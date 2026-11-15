"""Read-only GPU chain validation. No command buffers, transport, or robot I/O.

Predict the actual Y -> X -> fixed-orientation lift from the current joints.
Every phase uses the same continuous RT spline checks as execution. Predictions
are not execution authorization; the normal fresh-start planner checks remain.
"""
import threading
import time
from types import SimpleNamespace
import numpy as np

OPERATION='approach_lift_preflight_v1'
MAX_SECONDS=45.


def validate_request(model,request):
    q=np.asarray(request['joints'],float);grip=np.asarray(request['grippers'],float)
    target=np.asarray(request['target_xyz_m'],float);lift=float(request['lift_mm'])
    deadline=float(request['deadline'])
    if (q.shape!=(14,) or grip.shape!=(2,) or target.shape!=(3,)
            or not all(np.isfinite(x).all() for x in (q,grip,target))
            or not np.isfinite(lift) or not 0<lift<=600 or not np.isfinite(deadline)
            or deadline-time.monotonic()>MAX_SECONDS+1
            or np.any(q<model.bounds[0]) or np.any(q>model.bounds[1])
            or np.any(grip<0) or np.any(grip>1)):
        raise ValueError('Invalid approach/lift preflight input')
    xyz,_=model.tcp(q);initial=xyz[0,1]
    if (abs(target[2]-initial[2]-lift*.001)>.003
            or np.any(np.abs(target-initial)>.600001)):
        raise ValueError('Preflight target disagrees with measured lift or phase range')
    return q,grip,target,lift,deadline


def run_chain(model,q,grip,target,lift_mm,plan_phase):
    """Feed each validated trajectory endpoint into the next; never reuse initial q."""
    current=np.array(q,float,copy=True);phases=[]
    report=dict(accepted=False,operation=OPERATION,no_motion=True,
                start_joints_rad=current.tolist(),target_xyz_m=np.asarray(target).tolist(),
                lift_mm=float(lift_mm),phases=phases,
                limitation='predicted branches; execution replans from fresh measured joints')
    initial_xyz,initial_quat=model.tcp(current);quat=initial_quat[0,1].copy()
    for name,axis in [('approach_y',1),('advance_x',0),('lift_z',2)]:
        xyz,_=model.tcp(current);position=xyz[0,1].copy()
        if axis==2:position[2]+=lift_mm*.001
        else:position[axis]=target[axis]
        if abs(position[axis]-xyz[0,1,axis])<.0001:
            phases.append(dict(stage=name,accepted=True,skipped=True));continue
        end,detail=plan_phase(current,grip,position,quat,axis)
        detail=dict(detail,stage=name);phases.append(detail)
        if not detail.get('accepted'):
            report.update(failed_stage=name,reason=detail.get('reason','path_rejected'))
            return report
        end=np.asarray(end,float)
        if end.shape!=(14,) or not np.isfinite(end).all() or np.max(abs(end[:7]-q[:7]))>.002:
            raise ValueError('Preflight returned invalid endpoint or moved held left arm')
        current=end.copy()
    report.update(accepted=True,end_joints_rad=current.tolist())
    return report


def proxy_for(p,model,q,grip,position,quat,axis,deadline):
    """Private scratch attributes; never mutate the live planner's targets/commands."""
    xyz,wxyz=model.tcp(q);xyz=xyz[0].copy();wxyz=wxyz[0].copy()
    xyz[1]=position;wxyz[1]=quat
    def checkpoint():
        if time.monotonic()>=deadline:raise TimeoutError('Lift preflight GPU deadline exceeded')
    soft=np.asarray(p.arm_soft_joint_limits,float)
    bounds=np.stack([np.maximum(soft[0],model.bounds[0]),np.minimum(soft[1],model.bounds[1])])
    if np.any(q<bounds[0]) or np.any(q>bounds[1]):raise ValueError('Preflight start outside execution limits')
    proxy=SimpleNamespace(total_arm_joint_names=list(p.total_arm_joint_names),
        target_link_names=list(p.target_link_names),target_gripper_joint=model.grip_matrix@grip+model.grip_bias,
        arm_joint_position=q.copy(),arm_soft_joint_limits=bounds,_motion_gen=p._motion_gen,
        tensor_args=p.tensor_args,_v31_bridge=p._v31_bridge,_gpu_lock=threading.RLock(),
        target_description=('book_plug_x','book_plug_y','book_plug_z')[axis],
        target_arm_tool_position=xyz,target_arm_tool_orientation=wxyz,
        _trial_z_hold_weight=getattr(p,'_trial_z_hold_weight',20.),
        raise_if_state_thread_failed=checkpoint,_move_arm=[False,True])
    proxy.reorder_input=lambda arms,grips,as_curobo:np.r_[arms,grips][model.full_order]
    return proxy


def plan_phase(p,model,q,grip,position,quat,axis,deadline):
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from j6_geometry import rotation
    from book_lower_pose import right_link_hold
    from book_post_lift_pose import execution_bounds,validate_endpoint
    from continuous_trajectory import prepare
    from j6_trajectory import panel_path
    from lift_continuity import lift_start,source_metrics
    proxy=proxy_for(p,model,q,grip,position,quat,axis,deadline)
    names=list(p.total_arm_joint_names)+list(p.total_gripper_joint_names)
    initial=JointState.from_position(p.tensor_args.to_device(np.r_[q,proxy.target_gripper_joint][None]),joint_names=names)
    poses={}
    for i,name in enumerate(p.target_link_names):
        flange=proxy.target_arm_tool_position[i]-rotation(proxy.target_arm_tool_orientation[i]).apply(model.offsets[i])
        poses[name]=Pose(position=p.tensor_args.to_device(flange),quaternion=p.tensor_args.to_device(proxy.target_arm_tool_orientation[i]))
    config=(p._plan_config.clone() if hasattr(p,'_plan_config') else MotionGenPlanConfig(max_attempts=3,timeout=6.))
    config.timeout=min(6.,max(.01,deadline-time.monotonic()));config.max_attempts=min(3,config.max_attempts)
    detail=dict(accepted=False,axis='XYZ'[axis],start_joints_rad=q.tolist(),
                target_xyz_m=position.tolist(),target_wxyz=quat.tolist())
    began=time.monotonic()
    from lift_ik_diagnostics import capture
    with capture(p._motion_gen.ik_solver,p.total_arm_joint_names,q,detail),execution_bounds(proxy),right_link_hold(proxy,advance_axis=axis):
        result=p._motion_gen.plan_single(initial,poses[p.ee_link_name],config,link_poses=poses)
    detail.update(solver_status=str(result.status),solver_success=bool(result.success.item()))
    if not detail['solver_success']:
        detail.update(reason='local_lift_ik_failed' if axis==2 and 'IK_FAIL' in str(result.status) else 'gpu_path_failed',
                      lift_continuity=getattr(proxy,'_lift_continuity_diagnostics',None),elapsed_s=time.monotonic()-began)
        return None,detail
    raw=result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy().copy()
    result=prepare(proxy,result);detail['continuous']=proxy._continuous_diagnostics
    if not bool(result.success.item()):
        detail.update(reason='continuous_validation_failed',elapsed_s=time.monotonic()-began);return None,detail
    points=result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy()
    proxy.waypoints=points.copy();proxy.limited_waypoints=points.copy()
    proxy.waypoints_dt=float(result.optimized_dt.item())
    proxy._cmd_plan=result.get_interpolated_plan().get_ordered_joint_state(p.total_arm_joint_names)
    if not validate_endpoint(proxy):
        detail.update(reason='endpoint_or_lift_path_failed',endpoint=getattr(proxy,'_book_post_lift_diagnostics',None),
                      lift_continuity=getattr(proxy,'_lift_continuity_diagnostics',None));return None,detail
    if lift_start(proxy) is not None:
        panel=panel_path(proxy,points,proxy.waypoints_dt,q)
        metric=source_metrics(q,points,raw,panel,[False,True]);detail['lift_continuity']=metric
        if not metric['accepted']:
            detail.update(reason='lift_neighbourhood_path_failed');return None,detail
    proxy.raise_if_state_thread_failed()
    detail.update(accepted=True,elapsed_s=time.monotonic()-began,end_joints_rad=points[-1].tolist())
    return points[-1].copy(),detail


def solve(service,request):
    q,grip,target,lift,deadline=validate_request(service.model,request)
    began=time.monotonic();p=service.planner
    def phase(q,grip,position,quat,axis):
        remaining=deadline-time.monotonic()
        if remaining<=0 or not p._gpu_lock.acquire(timeout=max(0.,remaining)):
            return None,dict(accepted=False,reason='preflight_gpu_lock_timeout')
        try:
            if service.stopped.is_set():raise RuntimeError('Planner preparation service stopped')
            try:
                return plan_phase(p,service.model,q,grip,position,quat,axis,deadline)
            except (TimeoutError, ValueError) as error:
                return None,dict(accepted=False,reason=f'{type(error).__name__}: {error}',
                                 failure_source='readonly_path_check',axis='XYZ'[axis])
        finally:p._gpu_lock.release()
    proof=run_chain(service.model,q,grip,target,lift,phase)
    proof['elapsed_s']=time.monotonic()-began
    return dict(preflight=proof)


def request_preflight(model,report,target,lift_mm,*,seconds=MAX_SECONDS,reject=True):
    from start import existing_planner
    from j6_only import require_capability,revision
    from planning_receipt import process_stamp
    from gpu_preparation import socket_path,send_message,read_message
    import socket,uuid
    seconds=float(seconds)
    if not np.isfinite(seconds) or not 0<seconds<=MAX_SECONDS:
        raise ValueError('Invalid preflight time budget')
    pid=existing_planner();require_capability(pid);stamp=process_stamp(pid)
    request=dict(operation=OPERATION,request_id=uuid.uuid4().hex,planner_pid=pid,planner_start=stamp,
                 revision=revision(),model_digest=model.digest,joints=report['start_joints_rad'],
                 grippers=report['gripper_feedback'],target_xyz_m=np.asarray(target).tolist(),
                 lift_mm=float(lift_mm),deadline=time.monotonic()+seconds)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(seconds+1);sock.connect(str(socket_path(pid)));send_message(sock,request);reply=read_message(sock)
    for key in ('request_id','planner_pid','planner_start','revision'):
        if reply.get(key)!=request[key]:raise RuntimeError('Lift preflight identity mismatch: '+key)
    if process_stamp(pid)!=stamp:raise RuntimeError('Planner changed during lift preflight')
    if reply.get('error'):raise RuntimeError('Lift preflight failed: '+reply['error'])
    proof=reply['preflight']
    proof['identity']={k:request[k] for k in ('request_id','planner_pid','planner_start','revision','model_digest')}
    report['approach_lift_preflight']=proof
    if reject and not proof.get('accepted'):
        import json
        failed=next((item for item in proof.get('phases',[]) if not item.get('accepted')), {})
        best=failed.get('best_ik_candidate')
        suffix=('；较优IK候选='+json.dumps(best,ensure_ascii=False)) if best else ''
        raise ValueError('Approach前实际路径预检拒绝：阶段='+str(proof.get('failed_stage'))+
                         '，原因='+str(proof.get('reason'))+suffix+'；未执行Approach，详见approach_lift_preflight.phases')
    return proof
