"""Preparation-only GPU joint connection and exact RT curve validation.

Readonly RPC and actual execution use the same generator. No transport writes.
The execution caller still replans and validates from fresh measured joints.
"""
import time
import json
from types import SimpleNamespace
import numpy as np

ACTION = 'right_trial_j6_axis_align'
# Existing measured-start tolerance, distinct from exact held-joint path checks.
START_MATCH_RAD = .002


def bind_start_and_goal(input_start, measured_start, requested_goal, move_arm):
    """Bind inactive joints to one measured start, never command an old hold.

    Request-vs-feedback drift and commanded path drift are different quantities.
    The former has the existing start tolerance; the latter stays at 1e-6 rad.
    """
    incoming, q, requested = [np.asarray(x, float) for x in
                              (input_start, measured_start, requested_goal)]
    if (any(a.shape != (14,) for a in (incoming,q,requested))
            or not np.isfinite([incoming,q,requested]).all()):
        raise ValueError('Invalid preparation start/goal')
    if len(move_arm)!=2 or bool(move_arm[0]):
        raise ValueError('Preparation connection cannot enable left arm')
    input_delta=float(np.max(abs(incoming-q)))
    left_delta=requested[:7]-q[:7]
    if input_delta > START_MATCH_RAD:
        raise ValueError('Preparation input start changed beyond measured-start tolerance')
    if np.max(abs(left_delta)) > START_MATCH_RAD:
        raise ValueError('Preparation inactive left goal changed beyond measured-start tolerance')
    target=requested.copy();target[:7]=q[:7]
    detail=dict(policy='measured_start_and_inactive_left_hold',
        input_start_joints_rad=incoming.tolist(),measured_start_joints_rad=q.tolist(),
        requested_goal_joints_rad=requested.tolist(),effective_goal_joints_rad=target.tolist(),
        input_start_delta_rad=input_delta,inactive_left_delta_rad=left_delta.tolist(),
        max_inactive_left_delta_rad=float(np.max(abs(left_delta))),
        start_match_limit_rad=START_MATCH_RAD,commanded_left_hold_limit_rad=1e-6)
    return q.copy(),target,detail


def check_path(samples, start, bounds):
    a = np.asarray(samples, float)
    if (a.ndim != 2 or a.shape[1] != 14 or not len(a)
            or not np.isfinite(a).all()):
        raise ValueError('Invalid preparation connection samples')
    if np.any(a < bounds[0]) or np.any(a > bounds[1]):
        raise ValueError('Preparation connection joint limit')
    if np.max(abs(a[:, :7]-start[:7])) > 1e-6:
        raise ValueError('Preparation connection moved left arm')


def candidate(p, start, goal, deadline=None):
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from book_planning_budget import FailedPlan
    from continuous_trajectory import smooth_points, duration_candidates, COUNTS
    from j6_trajectory import limits_for, panel_path, collision_gate
    from wrist_timing import derivative_peaks
    began = time.monotonic()
    deadline = min(deadline or began+7., getattr(p, '_continuous_planning_deadline', began+7.))
    report = dict(accepted=False, no_motion=True, policy='GPU_joint_connection_then_RT_validation')
    def checkpoint():
        if time.monotonic() >= deadline:raise TimeoutError('preparation_connection_deadline')
        p.raise_if_state_thread_failed()
    try:
        names = list(p.total_arm_joint_names)
        incoming = start.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
        requested = goal.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
        report.update(input_start_joints_rad=incoming.tolist(),requested_goal_joints_rad=requested.tolist(),
                      measured_start_joints_rad=np.asarray(p.arm_joint_position).tolist())
        q,target,binding=bind_start_and_goal(incoming,p.arm_joint_position,requested,p._move_arm)
        report['start_binding']=binding
        # Build both solver inputs from the same latched measurement. Keep the
        # right goal exactly requested and the grippers at their original hold.
        full=start.position.detach().cpu().numpy().reshape(1,-1).copy()
        ordered_goal=goal.get_ordered_joint_state(start.joint_names).position.detach().cpu().numpy().reshape(1,-1)
        inactive=[i for i,n in enumerate(start.joint_names) if n not in names]
        if inactive and np.max(abs(full[:,inactive]-ordered_goal[:,inactive]))>1e-6:
            raise ValueError('Preparation connection cannot change grippers')
        for i,n in enumerate(names):full[0,start.joint_names.index(n)]=q[i]
        goal_full=full.copy()
        for i,n in enumerate(names):goal_full[0,start.joint_names.index(n)]=target[i]
        start=JointState.from_position(p.tensor_args.to_device(full),joint_names=list(start.joint_names))
        goal=JointState.from_position(p.tensor_args.to_device(goal_full),joint_names=list(start.joint_names))
        bounds, dynamics = limits_for(p)
        check_path(np.array([q, target]), q, bounds)
        checkpoint()
        config = MotionGenPlanConfig(max_attempts=3, timeout=min(4., deadline-time.monotonic()))
        with p._gpu_lock:
            raw = p._motion_gen.plan_single_js(start, goal, config)
        checkpoint()
        report['solver_status'] = str(raw.status)
        if not bool(raw.success.item()):raise ValueError('preparation_connection_GPU_rejected: '+str(raw.status))
        source = raw.optimized_plan.get_ordered_joint_state(names).position.detach().cpu().numpy().copy()
        # Left solver redundancy is never commanded; held-left curve is checked anew.
        source[:, :7] = q[:7]
        if np.max(abs(source[-1]-target)) > .002:
            raise ValueError('Preparation connection endpoint mismatch')
        # Exact goal, including right J7; all changes are checked on the RT curve.
        source[-1] = target
        accepted = None
        attempts = []
        # Keep rejected dynamics evidence too; it used to disappear on failure.
        report['attempts']=attempts
        for count in COUNTS:
            points = smooth_points(source, q, count).astype(np.float32).astype(float)
            check_path(points, q, bounds)
            for dt in duration_candidates(float(raw.optimized_dt.item()), len(source), count):
                checkpoint()
                panel = panel_path(p, points, dt, q)
                check_path(panel, q, bounds)
                peaks = derivative_peaks(panel, dynamics)
                ratios = [x['ratio'] for x in peaks]
                attempts.append(dict(count=count,dt=dt,ratios=ratios))
                if max(ratios)>1.:continue
                collision_gate(p, np.concatenate([q[None],points,panel]), checkpoint)
                checkpoint()
                accepted = points,panel,dt,ratios
                break
            if accepted is not None:break
        if accepted is None:raise ValueError('Preparation connection RT timing rejected')
        points,panel,dt,ratios = accepted
        def state(values, spacing):
            full = np.repeat(start.position.detach().cpu().numpy().reshape(1,-1),len(values),axis=0)
            for k,name in enumerate(names):full[:,start.joint_names.index(name)] = values[:,k]
            vel=np.gradient(full,spacing,axis=0);acc=np.gradient(vel,spacing,axis=0);jerk=np.gradient(acc,spacing,axis=0)
            return JointState.from_numpy(list(start.joint_names),full,vel,acc,jerk,tensor_args=p.tensor_args)
        result = SimpleNamespace(success=np.array(True),status='PREPARATION_CONNECTION_VALIDATED',
            optimized_plan=state(points,dt),optimized_dt=p.tensor_args.to_device([dt]),
            ik_time=0.,graph_time=0.,trajopt_time=0.,finetune_time=0.)
        command=panel[np.r_[np.arange(0,len(panel)-1,20),len(panel)-1]]
        result.get_interpolated_plan=lambda:state(command,.02)
        result.solve_time=result.total_time=time.monotonic()-began
        result.validated_joint_trim_steps=(0,len(points))
        report.update(accepted=True,start_joints_rad=q.tolist(),goal_joints_rad=target.tolist(),
            samples=len(panel),duration_s=(len(panel)-1)*.001,dynamics_ratios=ratios,attempts=attempts)
    except Exception as error:
        report['reason']=f'{type(error).__name__}: {error}'
        result=FailedPlan(report['reason'],time.monotonic()-began)
    report['elapsed_s']=time.monotonic()-began
    result.preparation_connection=report
    if hasattr(p, '_deadline_report'):
        p._deadline_report['preparation_connection']=report
    print('Preparation connection: '+json.dumps(report,allow_nan=False),flush=True)
    return result


def validate_before_publish(p, result):
    from j6_trajectory import limits_for,panel_path,collision_gate
    from wrist_timing import derivative_peaks
    proof=result.preparation_connection
    if not proof.get('accepted'):raise ValueError('Missing preparation path proof')
    q=np.asarray(p.arm_joint_position,float)
    if np.max(abs(q-np.asarray(proof['start_joints_rad'])))>.002:
        raise ValueError('Preparation connection start changed')
    points=result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy()
    if tuple(result.validated_joint_trim_steps)!=(0,len(points)):
        raise ValueError('Preparation connection trim changed')
    bounds,dynamics=limits_for(p)
    panel=panel_path(p,points,float(result.optimized_dt.item()),q)
    check_path(np.concatenate([points,panel]),q,bounds)
    if np.max(abs(points[-1]-np.asarray(proof['goal_joints_rad'])))>.002:
        raise ValueError('Preparation connection goal changed')
    if max(x['ratio'] for x in derivative_peaks(panel,dynamics))>1.:
        raise ValueError('Preparation connection dynamics changed')
    from planning_deadline import checkpoint
    collision_gate(p,np.concatenate([q[None],points,panel]),lambda:checkpoint(p))
    checkpoint(p)


def serve(service, request):
    from curobo.types.state import JointState
    from lift_preflight import proxy_for
    p,model=service.planner,service.model
    q=np.asarray(request['joints'],float);target=np.asarray(request['goal_joints'],float)
    grip=np.asarray(request['grippers'],float);deadline=float(request['deadline'])
    if (q.shape!=(14,) or target.shape!=(14,) or grip.shape!=(2,)
            or not np.isfinite(grip).all() or np.any(grip<0) or np.any(grip>1)
            or not np.isfinite(deadline) or deadline-time.monotonic()>9.):
        raise ValueError('Invalid connection request')
    _,effective,_=bind_start_and_goal(q,q,target,[False,True])
    check_path(np.array([q,effective]),q,model.bounds)
    remaining=deadline-time.monotonic()
    if remaining<=0 or not p._gpu_lock.acquire(timeout=remaining):return dict(expired=True)
    try:
        xyz,quat=model.tcp(q)
        proxy=proxy_for(p,model,q,grip,xyz[0,1],quat[0,1],0,deadline)
        proxy.target_description=ACTION
        proxy._continuous_planning_deadline=deadline
        names=list(p.total_arm_joint_names)+list(p.total_gripper_joint_names)
        def state(joints):return JointState.from_position(p.tensor_args.to_device(np.r_[joints,proxy.target_gripper_joint][None]),joint_names=names)
        result=candidate(proxy,state(q),state(target),deadline)
        return dict(diagnostics=result.preparation_connection)
    finally:p._gpu_lock.release()
