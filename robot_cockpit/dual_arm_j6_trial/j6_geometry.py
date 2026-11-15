"""Base-Y J6 alignment and a measured-start, single-joint downward sweep."""
import time
import numpy as np
from scipy.spatial.transform import Rotation

AXIS_LIMIT_DEG = 1.0
SWEEP_DEG = 31.0
# User observed camera-down at finger_base=-Y in run 20261113_135751.
# Select the 180-degree-equivalent +Y branch; this is an installation convention,
# not a calibrated camera optical transform.
CAMERA_UP_FINGER_Y_SIGN = 1.0


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1,2,3,0]])


def axis_base(model, joints):
    q = np.asarray(joints, float)
    probe = q.copy(); probe[12] += 1e-3
    _, r0 = model.tcp(q); _, r1 = model.tcp(probe)
    axis = (rotation(r1[0,1])*rotation(r0[0,1]).inv()).as_rotvec()/1e-3
    return axis/np.linalg.norm(axis)


def axis_error(model, joints):
    return float(np.rad2deg(np.arccos(np.clip(abs(axis_base(model,joints)[1]),-1,1))))


def collision_free(model, joints, grippers):
    grip = model.grip_matrix @ np.asarray(grippers,float) + model.grip_bias
    q = np.atleast_2d(joints)
    ordered = np.concatenate([q,np.broadcast_to(grip,(len(q),len(grip)))],axis=1)[:,model.full_order]
    return model.collision.check(ordered)


def downward_goal(model, joints, grippers=None, *, require_fingers=True):
    q = np.asarray(joints,float)
    error = axis_error(model,q)
    if error > AXIS_LIMIT_DEG:
        raise ValueError(f'J6轴与基座+Y偏差{error:.3f}°，超过{AXIS_LIMIT_DEG}°；不能仅J6下拨')
    if require_fingers:check_finger_alignment(model,q)
    goal = q.copy(); goal[12] += np.sign(axis_base(model,q)[1])*np.deg2rad(SWEEP_DEG)
    if np.any(goal<model.bounds[0]) or np.any(goal>model.bounds[1]):
        raise ValueError(f'J6向下{SWEEP_DEG:g}°目标超出模型关节限位')
    samples = np.repeat(q[None,:],61,axis=0)
    samples[:,12] = np.linspace(q[12],goal[12],61)
    p, r = model.tcp(samples)
    displacement = (p[-1,1]-p[0,1])*1000
    tool = np.array(model.offsets[1],dtype=float,copy=True); tool /= np.linalg.norm(tool)
    forward = rotation(r[0,1]).apply(tool)
    if forward[0] < .95 or displacement[2] >= 0 or displacement[0] >= 0:
        raise ValueError('J6曲线未实现朝前夹爪向下、向机器人拨动；停止')
    if grippers is not None and not collision_free(model,samples,grippers):
        raise ValueError('J6下拨采样路径存在模型碰撞或限位冲突')
    return goal, dict(axis_base=axis_base(model,q).tolist(),axis_error_deg=error,
                     joint_delta_deg=float(np.rad2deg(goal[12]-q[12])), tcp_displacement_mm=displacement.tolist(),
                     sampled_tcp_mm=(p[:,1]*1000).tolist(), fixed_joint_count=13)


ALIGNMENT_SEARCH_SECONDS = 8.0


class _AlignmentBudgetExpired(Exception):
    pass


class AlignmentRejected(ValueError):
    def __init__(self, summary):
        self.summary=summary
        super().__init__('无法在指定TCP处求得满足轴向/位置约束的J6准备姿态（不预留下拨角度）：'+str(summary))


def alignment_jobs(q, bounds):
    """Try nearby wrists first, then reproducible shoulder/elbow/wrist branches."""
    rng = np.random.default_rng(521)
    for phase in ('near', 'branch'):
        for sign in (1., -1.):
            lo = bounds[0,7:].copy()+np.deg2rad(2.)
            hi = bounds[1,7:].copy()-np.deg2rad(2.)
            if np.any(lo >= hi):
                raise ValueError('准备姿态关节搜索区间为空')
            for index in range(3 if phase == 'near' else 12):
                if phase == 'near':
                    shift = (0., .5, -.5)[index]
                    seed = q[7:]+np.array([0,0,0,0,shift,0,-shift])
                else:
                    seed = lo+(hi-lo)*rng.uniform(.02,.98,7)
                    for joint in (0,2,4,6):
                        seed[joint] = q[7+joint]+rng.uniform(-np.pi,np.pi)
                yield sign, lo, hi, np.clip(seed,lo+1e-6,hi-1e-6), f'{phase}_{sign:+g}_{index}'


def preparation_orientation(model, joints, sign):
    """Exact forward/+Y (or -Y) orientation with J7 held for this IK solve."""
    q=np.asarray(joints,float)
    _,quat=model.tcp(q)
    forward=np.array(model.offsets[1],dtype=float,copy=True)
    forward/=np.linalg.norm(forward)
    local_axis=rotation(quat[0,1]).inv().apply(axis_base(model,q))
    if abs(float(forward@local_axis))>1e-5:
        raise ValueError('J6 axis/tool geometry is not orthogonal; cannot construct preparation pose')
    local_axis-=forward*(forward@local_axis);local_axis/=np.linalg.norm(local_axis)
    local=np.column_stack((forward,local_axis,np.cross(forward,local_axis)))
    x=np.array([1.,0.,0.]);y=np.array([0.,sign,0.])
    base=np.column_stack((x,y,np.cross(x,y)))
    xyzw=Rotation.from_matrix(base@local.T).as_quat()
    return xyzw[[3,0,1,2]]


def align_goal(model, joints, grippers, target_position_m=None, *, search_seconds=None,
               quick=False, preferred_joints=None):
    """CuRobo GPU IK generates candidates; CPU FK only verifies constraints."""
    from gpu_preparation import solve_candidates
    q=np.asarray(joints,float)
    if q.shape!=(14,) or not np.isfinite(q).all():raise ValueError('Invalid preparation joints')
    p0,_=model.tcp(q)
    target_p=p0[0,1] if target_position_m is None else np.asarray(target_position_m,float)
    if target_p.shape!=(3,) or not np.isfinite(target_p).all():raise ValueError('Invalid target TCP')
    tool=np.array(model.offsets[1],dtype=float,copy=True);tool/=np.linalg.norm(tool)
    began=time.monotonic();failures=[];attempts=0
    budget=ALIGNMENT_SEARCH_SECONDS if search_seconds is None else min(float(search_seconds),ALIGNMENT_SEARCH_SECONDS)
    if not np.isfinite(budget):raise ValueError('Invalid preparation search budget')
    jobs=list(alignment_jobs(q,model.bounds))
    if quick:jobs=[j for j in jobs if j[4] in ('near_+1_0','near_-1_0')]
    if preferred_joints is not None:
        preferred=np.asarray(preferred_joints,float)
        if preferred.shape!=(14,) or not np.isfinite(preferred).all():raise ValueError('Invalid preferred preparation joints')
        _,lo,hi,_,_=jobs[0]
        jobs.insert(0,(float(np.sign(axis_base(model,preferred)[1])) or 1.,lo,hi,
                       np.clip(preferred[7:],lo+1e-6,hi-1e-6),'verified_preparation_branch'))
    # Explicit orientation targets inside the existing 1-degree acceptance cone.
    # This avoids treating an exact 0-degree pose as a new, stricter constraint.
    jobs=[(*job,yaw) for yaw in (0.,-.5,.5,-.9,.9) for job in jobs]
    exhausted=False
    for sign,lo,hi,seed,name,yaw in jobs:
        remaining=budget-(time.monotonic()-began)
        if remaining<=0:exhausted=True;break
        quat=preparation_orientation(model,q,sign)
        xyzw=(Rotation.from_euler('z',yaw,degrees=True)*rotation(quat)).as_quat()
        quat=xyzw[[3,0,1,2]]
        name=name+f'_yaw{yaw:+g}'
        candidates,gpu=solve_candidates(model,q,grippers,target_p,quat,np.r_[q[:7],seed],remaining)
        attempts+=1
        if gpu.get('budget_exhausted'):exhausted=True;break
        if not candidates:
            failures.append(dict(seed=name,solver='CuRobo GPU IK',rejection_reasons=['gpu_ik_no_solution'],gpu=gpu))
        for values in candidates:
            goal=np.asarray(values,float)
            if goal.shape!=(14,) or not np.isfinite(goal).all():raise ValueError('Invalid GPU IK candidate')
            p,r=model.tcp(goal)
            pe=float(np.linalg.norm(p[0,1]-target_p)*1000)
            ae=axis_error(model,goal)
            oe=float(np.rad2deg(np.arccos(np.clip(rotation(r[0,1]).apply(tool)[0],-1,1))))
            checks=dict(position=bool(pe<=.5),axis=bool(ae<=AXIS_LIMIT_DEG),
                        tool_forward=bool(oe<=AXIS_LIMIT_DEG),
                        joint_limits=bool(np.all(goal[7:]>=lo) and np.all(goal[7:]<=hi)),
                        held_left_and_j7=bool(np.array_equal(goal[:7],q[:7]) and goal[13]==q[13]),
                        endpoint_collision_free=bool(collision_free(model,goal,grippers)))
            detail=dict(seed=name,solver='CuRobo GPU IK',solver_success=True,gpu=gpu,
                        search_attempt=attempts,search_elapsed_s=time.monotonic()-began,
                        position_error_mm=pe,axis_error_deg=ae,tool_forward_error_deg=oe,
                        before_axis_base=axis_base(model,q).tolist(),after_axis_base=axis_base(model,goal).tolist(),
                        j6_start_deg=float(np.rad2deg(goal[12])),sweep_considered=False,
                        limits=dict(position_mm=.5,axis_deg=AXIS_LIMIT_DEG,tool_forward_deg=AXIS_LIMIT_DEG,joint_margin_deg=2.),
                        checks=checks,rejection_reasons=[k for k,v in checks.items() if not v],following_path_check='not_run')
            if all(checks.values()):
                try:j7_align_goal(model,goal,grippers)
                except ValueError as error:
                    detail.update(following_path_check='rejected',following_path_error=str(error))
                    detail['rejection_reasons'].append('j7_alignment')
                else:
                    detail['following_path_check']='passed_j7_path_only'
                    return goal,detail
            failures.append(detail)
    ranked=sorted(failures,key=lambda d:max(d.get('position_error_mm',float('inf'))/.5,
                  d.get('axis_error_deg',float('inf'))/AXIS_LIMIT_DEG,d.get('tool_forward_error_deg',float('inf'))/AXIS_LIMIT_DEG))
    raise AlignmentRejected(dict(attempts=attempts,budget_exhausted=exhausted,elapsed_s=time.monotonic()-began,
        best_candidates=ranked[:6],solver='CuRobo GPU IK',policy='GPU_solve_independent_FK_checks_no_CPU_solver_fallback'))


def finger_axis_base(model,joints):
    """URDF opening direction (line joining the fingers), not finger length."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    # Fixed flange->gripper and prismatic joint-frame rotation define opening axis.
    tree=ET.parse(Path(__file__).parent/'configuration/dual_gripper.urdf')
    base=tree.find(".//joint[@name='gripper_base_joint_r']/origin")
    joint=tree.find(".//joint[@name='gripper_J1_r']")
    a=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
    local=(Rotation.from_euler('xyz',np.fromstring(base.get('rpy','0 0 0'),sep=' '))*
           Rotation.from_euler('xyz',np.fromstring(joint.find('origin').get('rpy','0 0 0'),sep=' '))).apply(a)
    _,quat=model.tcp(joints)
    return rotation(quat[0,1]).apply(local/np.linalg.norm(local))


def check_finger_alignment(model,joints):
    finger=finger_axis_base(model,joints)
    error=float(np.rad2deg(np.arccos(np.clip(CAMERA_UP_FINGER_Y_SIGN*finger[1],-1.,1.))))
    relative=float(np.rad2deg(np.arccos(np.clip(abs(finger@axis_base(model,joints)),-1.,1.))))
    if error>AXIS_LIMIT_DEG or relative>AXIS_LIMIT_DEG:
        raise ValueError(f'双指未处于相机朝上分支或未平行基座Y/J6轴：base_Y={error:.3f}deg, J6={relative:.3f}deg')
    return dict(finger_axis_base=finger.tolist(),finger_base_y_error_deg=error,finger_j6_error_deg=relative, camera_up_branch=True,
                camera_orientation_basis='user_observed_opposite_of_20261113_135751',
                required_finger_y_sign=CAMERA_UP_FINGER_Y_SIGN)


def j7_align_goal(model,joints,grippers=None):
    q=np.asarray(joints,float)
    if axis_error(model,q)>AXIS_LIMIT_DEG:raise ValueError('先对齐J6轴，再仅J7调整双指')
    probe=q.copy();probe[13]+=.001
    _,r0=model.tcp(q);_,r1=model.tcp(probe)
    axis=(rotation(r1[0,1])*rotation(r0[0,1]).inv()).as_rotvec()/.001
    axis/=np.linalg.norm(axis)
    finger=finger_axis_base(model,q)
    source=finger-axis*(finger@axis);source/=np.linalg.norm(source)
    target=np.array([0.,1.,0.]);target-=axis*(target@axis)
    if np.linalg.norm(target)<.9:raise ValueError('仅J7无法使双指平行基座Y')
    target/=np.linalg.norm(target)
    angle=np.arctan2(axis@np.cross(source,target),source@target)
    candidates=sorted([angle+k*np.pi for k in range(-2,3)],key=abs)
    errors=[]
    for delta in candidates:
        if abs(delta)>np.pi+1e-8:continue
        goal=q.copy();goal[13]+=delta
        if np.any(goal<model.bounds[0]) or np.any(goal>model.bounds[1]):continue
        try:
            detail=check_finger_alignment(model,goal)
            samples=np.repeat(q[None,:],max(2,int(np.ceil(abs(np.rad2deg(delta))))+1),axis=0)
            samples[:,13]=np.linspace(q[13],goal[13],len(samples))
            if grippers is not None and not collision_free(model,samples,grippers):
                raise ValueError('J7双指调整路径碰撞')
        except ValueError as e:errors.append(str(e));continue
        detail.update(joint_delta_deg=float(np.rad2deg(delta)),moving_joint='right J7',
                      finger_axis_before=finger.tolist(),fixed_joint_count=13,
                      branch_policy='camera_up_signed_Y_then_shortest_safe_path')
        return goal,detail
    raise ValueError('没有限位内且无碰撞的仅J7双指对齐路径：'+str(errors))
