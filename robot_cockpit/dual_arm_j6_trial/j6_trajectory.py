"""One-dimensional J6 or J7 curve, validated with the deployed panel interpolator.

No sockets or commands: returns a candidate to the ordinary publication path.
"""
import json
import time
from types import SimpleNamespace
import numpy as np
from j6_only import HOLD_RAD, moving_joint
from j6_geometry import SWEEP_DEG

SAMPLE_DT = .001
SPEED_FACTOR = 0.5


def points_for(start, goal, index, count):
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    if (start.shape!=(14,) or goal.shape!=(14,) or not np.isfinite([start,goal]).all()
            or index not in (12,13) or not 6<=count<=128):
        raise ValueError('Invalid J6 start/goal/waypoint count')
    fixed=[i for i in range(14) if i!=index]
    if np.max(np.abs(goal[fixed]-start[fixed]))>HOLD_RAD:
        raise ValueError('J6 request changes other joints beyond holding tolerance')
    delta=goal[index]-start[index]
    if index==12 and abs(abs(delta)-np.deg2rad(SWEEP_DEG))>np.deg2rad(.5):
        raise ValueError(f'J6 request must be a {SWEEP_DEG:g}-degree rotation')
    if index==13 and abs(delta)>np.pi+1e-8:raise ValueError('J7 preparation rotation exceeds 180deg')
    t=np.linspace(0.,1.,count)
    blend=10*t**3-15*t**4+6*t**5
    points=np.repeat(start[None,:],count,axis=0)
    points[:,index]=start[index]+delta*blend
    points[0,index]=start[index];points[-1,index]=goal[index]
    return points


def check_samples(samples,start,index,bounds,*,exact_fixed=False):
    samples,start,bounds=np.asarray(samples,float),np.asarray(start,float),np.asarray(bounds,float)
    if (samples.ndim!=2 or samples.shape[1]!=14 or not len(samples) or start.shape!=(14,)
            or bounds.shape!=(2,14) or not np.isfinite(samples).all() or not np.isfinite(bounds).all()
            or np.any(bounds[0]>=bounds[1])):
        raise ValueError('Invalid J6 path or position limits')
    if np.any(samples<bounds[0]) or np.any(samples>bounds[1]):
        raise ValueError('J6 path exceeds joint position limits')
    fixed=[i for i in range(14) if i!=index]
    drift=float(np.max(np.abs(samples[:,fixed]-start[fixed])))
    if drift>(1e-6 if exact_fixed else HOLD_RAD):
        raise ValueError(f'J6 path moves a held joint: {drift:g} rad')
    return drift


def ratios(samples,limits):
    derivative=np.pad(samples,((3,3),(0,0)),mode='edge')
    values=[]
    for limit in limits:
        derivative=np.diff(derivative,axis=0)/SAMPLE_DT
        values.append(float(np.max(np.abs(derivative)/limit)))
    return np.array(values)


def panel_path(p,points,dt,start):
    paths=[]
    for a in range(2):
        path,_=p._v31_bridge.panel_path.sample(start[a*7:a*7+7],points[:,a*7:a*7+7],
                         dt,p._v31_bridge.limits[a],sample_dt=SAMPLE_DT)
        paths.append(path)
    n=max(map(len,paths))
    return np.concatenate([np.pad(x,((0,n-len(x)),(0,0)),mode='edge') for x in paths],axis=1)


def collision_gate(p,samples,checkpoint):
    grip=list(p.target_gripper_joint)
    order=np.asarray(p.reorder_input(np.arange(14),list(range(14,14+len(grip))),as_curobo=True),int)
    values=np.concatenate([samples,np.broadcast_to(grip,(len(samples),len(grip)))],axis=1)[:,order]
    with p._gpu_lock:
        for i in range(0,len(values),512):
            checkpoint()
            metrics=p._motion_gen.rollout_fn.rollout_constraint(
                p.tensor_args.to_device(np.ascontiguousarray(values[i:i+512])).unsqueeze(1),use_batch_env=False)
            if not bool(metrics.feasible.all().item()):
                raise ValueError(f'J6 path collision/model constraint failed in sample block {i}:{min(i+512,len(values))}')


def limits_for(p):
    from baseline_withdraw_path import dynamics_limits
    with p._gpu_lock:
        dynamics=dynamics_limits(p)
        limits=p._motion_gen.kinematics.get_joint_limits()
        order=[limits.joint_names.index(n) for n in p.total_arm_joint_names]
        model=limits.position.detach().cpu().numpy()[:,order]
    soft=np.asarray(p.arm_soft_joint_limits,float)
    if soft.shape!=(2,14):raise ValueError('Missing J6 position limits')
    bounds=np.stack([np.maximum(model[0],soft[0]),np.minimum(model[1],soft[1])])
    return bounds,dynamics


def verify_axis_and_direction(q,target,index=12):
    from pathlib import Path
    from backend.board_servo.model import Model
    from j6_geometry import downward_goal, axis_error, check_finger_alignment, AXIS_LIMIT_DEG
    model=Model(Path(__file__).parent/'backend/board_servo')
    if index==13:
        if axis_error(model,q)>AXIS_LIMIT_DEG:raise ValueError('J7 preparation requires aligned J6 axis')
        detail=check_finger_alignment(model,target)
        return detail
    goal,detail=downward_goal(model,q)
    if abs(target[12]-goal[12])>np.deg2rad(.5):
        raise ValueError(f'J6目标方向/{SWEEP_DEG:g}度与基座Y向下拨动不一致')
    return detail


def candidate(p,start,goal):
    from curobo.types.state import JointState
    from book_planning_budget import FailedPlan
    began=time.monotonic()
    report=dict(event='single_wrist_joint_trajectory',action=p.target_description,accepted=False,speed_factor=SPEED_FACTOR,retiming=[])
    def checkpoint():
        if time.monotonic()-began>5.:raise ValueError('J6 candidate validation budget exhausted')
        p.raise_if_state_thread_failed()
    try:
        names=list(p.total_arm_joint_names);index=names.index(moving_joint(p.target_description))
        q=start.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
        target=goal.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
        report['axis_geometry']=verify_axis_and_direction(q,target,index)
        # This curve already contains its true start/end. Generic MotionGen
        # padding trim (normally [2,-2]) must not remove those endpoints.
        report['generic_trim_steps']=list(p.trim_steps)
        bounds,dynamics=limits_for(p)
        report.update(start_joints_rad=q.tolist(),target_joints_rad=target.tolist(),
                      dynamics_limits=dynamics.tolist())
        from wrist_timing import select_curve
        points,panel,dt,ratio=select_curve(q,target,index,dynamics,SPEED_FACTOR,points_for,
            lambda curve,spacing: panel_path(p,curve,spacing,q),
            lambda samples: check_samples(samples,q,index,bounds,exact_fixed=True),
            checkpoint,report)
        collision_gate(p,np.concatenate([points,panel]),checkpoint)
        checkpoint()
        def state(values,spacing):
            # Match CuRobo active joint ordering, retaining gripper entries.
            full=np.repeat(start.position.detach().cpu().numpy().reshape(1,-1),len(values),axis=0)
            for k,name in enumerate(names):full[:,start.joint_names.index(name)]=values[:,k]
            vel=np.gradient(full,spacing,axis=0);acc=np.gradient(vel,spacing,axis=0);jerk=np.gradient(acc,spacing,axis=0)
            return JointState.from_numpy(list(start.joint_names),full,vel,acc,jerk,tensor_args=p.tensor_args)
        result=SimpleNamespace(success=np.array(True),status=SimpleNamespace(name='SINGLE_WRIST_JOINT_VALIDATED'),
            optimized_plan=state(points,dt),optimized_dt=p.tensor_args.to_device([dt]),
            ik_time=0.,graph_time=0.,trajopt_time=0.,finetune_time=0.)
        # Dense samples describe the RT curve; waypoints remain exactly J6-only.
        command=panel[np.r_[np.arange(0,len(panel)-1,20),len(panel)-1]]
        result.get_interpolated_plan=lambda:state(command,.02)
        result.solve_time=result.total_time=time.monotonic()-began
        report.update(accepted=True,moving_joint=names[index],fixed_joint_count=13,
                      waypoint_count=len(points),waypoint_dt=dt,planned_duration_s=(len(panel)-1)*SAMPLE_DT,
                      dynamics_ratios=ratio.tolist(),fixed_joint_max_delta_rad=float(np.max(np.abs(points[:,np.arange(14)!=index]-q[np.arange(14)!=index]))))
        result.single_j6_diagnostics=report
        result.validated_joint_trim_steps=(0,len(points))
        return result
    except Exception as error:
        report['reason']=f'{type(error).__name__}: {error}'
        return FailedPlan('single_j6_rejected: '+report['reason'],time.monotonic()-began)
    finally:
        report['elapsed_s']=time.monotonic()-began
        print('Single wrist joint trajectory: '+json.dumps(report,allow_nan=False),flush=True)


def validate_before_publish(p,result):
    """Reconstruct the exact publication curve from the current measured start."""
    began=time.monotonic()
    def checkpoint():
        if time.monotonic()-began>3.:raise ValueError('J6 final validation budget exhausted')
        p.raise_if_state_thread_failed()
    names=list(p.total_arm_joint_names);index=names.index(moving_joint(p.target_description))
    points=result.optimized_plan.get_ordered_joint_state(names).position.detach().cpu().numpy()
    if tuple(result.validated_joint_trim_steps)!=(0,len(points)):
        raise ValueError('J6 publication must preserve the entire validated curve')
    actual=np.asarray(p.arm_joint_position,float).reshape(14)
    effective=np.array([p._clip_arm_joint_command(p._fix_non_moving_arm(q,actual)) for q in points])
    if not np.allclose(effective,points,atol=1e-6,rtol=0):raise ValueError('J6 candidate changed by clipping/fixed-arm publication')
    verify_axis_and_direction(actual,effective[-1],index)
    bounds,dynamics=limits_for(p)
    panel=panel_path(p,effective,float(result.optimized_dt.item()),actual)
    check_samples(panel,actual,index,bounds)
    from wrist_timing import derivative_peaks
    peaks=derivative_peaks(panel,dynamics)
    r=np.array([item['ratio'] for item in peaks])
    if np.any(r>1.):raise ValueError('J6 final splice exceeds velocity/acceleration/jerk limits: '+str(peaks))
    collision_gate(p,np.concatenate([actual[None,:],effective,panel]),checkpoint)
    checkpoint()
    print('Single wrist joint final path: '+json.dumps(dict(accepted=True,samples=len(panel),dynamics_ratios=r.tolist())),flush=True)
