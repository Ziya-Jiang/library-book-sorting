"""GPU refinement of forward X knots before one fully checked continuous RT curve.

No execution subdivision. Targets are a straight base-X line at fixed YZ/rotation;
nearest feasible IK seeds maintain the measured local joint branch. Full RT collision,
limits, dynamics, endpoint and orientation validation still decides publication.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation


def is_advance(p):
    if p.target_description != 'book_plug_x':return False
    xyz,_=p._v31_bridge.tcp(np.asarray(p.arm_joint_position))
    return float(p.target_arm_tool_position[1,0]-xyz[0,1,0]) > .001


def refine(p,result,check=lambda:None):
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from book_post_lift_pose import execution_bounds
    from planning_deadline import checkpoint
    from pathlib import Path
    from backend.board_servo.model import Model
    began=time.monotonic();model=Model(Path(__file__).parent/'backend/board_servo')
    arm_names=list(p.total_arm_joint_names);start=np.asarray(p.arm_joint_position,float).copy()
    xyz,wxyz=model.tcp(start);xyz,wxyz=xyz[0],wxyz[0]
    target=np.asarray(p.target_arm_tool_position[1]);quat=np.asarray(p.target_arm_tool_orientation[1])
    solver=p._motion_gen.ik_solver;names=list(solver.joint_names)
    full=p._motion_gen.get_full_js(result.optimized_plan)
    original=full.position.detach().cpu().numpy().reshape(-1,len(full.joint_names))
    template=dict(zip(full.joint_names,original[0]));values=dict(template);values.update(zip(arm_names,start))
    points=[np.array([values[n] for n in names])];details=[]
    # At most 10 mm between GPU IK targets; still only ONE published trajectory.
    distance=abs(target[0]-xyz[1,0])
    count=max(16,int(np.ceil(distance/.010))+1)
    if count>64:raise ValueError('X refinement needs more than 64 internal knots')
    with p._gpu_lock,execution_bounds(p):
        for t in np.linspace(0.,1.,count)[1:]:
            checkpoint(p);check()
            position=target.copy();position[0]=xyz[1,0]+t*(target[0]-xyz[1,0])
            poses={}
            for i,link in enumerate(p.target_link_names):
                pos,rot=(xyz[i],wxyz[i]) if i==0 else (position,quat)
                flange=pos-Rotation.from_quat(rot[[1,2,3,0]]).apply(model.offsets[i])
                poses[link]=Pose(position=p.tensor_args.to_device(flange[None]),quaternion=p.tensor_args.to_device(rot[None]))
            seed=p.tensor_args.to_device(points[-1][None])
            ik=solver.solve_single(poses[p.target_link_names[0]],seed_config=seed[:,None,:],
                retract_config=seed,return_seeds=solver.num_seeds,link_poses=poses)
            rows=ik.solution.detach().cpu().numpy().reshape(-1,len(names))
            success=ik.success.detach().cpu().numpy().reshape(-1)
            candidates=[]
            indices=[names.index(n) for n in arm_names]
            for row,ok in zip(rows,success):
                if not ok or not np.isfinite(row).all():continue
                q=row[indices].copy();q[:7]=start[:7]
                if np.max(abs(q[7:]-points[-1][indices][7:]))>np.deg2rad(10):continue
                qxyz,qq=model.tcp(q)
                error=np.linalg.norm(qxyz[0,1]-position)*1000
                angle=np.rad2deg((Rotation.from_quat(quat[[1,2,3,0]]).inv()*Rotation.from_quat(qq[0,1,[1,2,3,0]])).magnitude())
                if error>1 or angle>.25:continue
                row=row.copy();row[indices]=q
                for name in names:
                    if name not in arm_names:row[names.index(name)]=template[name]
                candidates.append((np.linalg.norm(row-points[-1]),row,error,angle))
            if not candidates:raise ValueError(f'X advance GPU refinement: no local valid knot at fraction {t:.3f}')
            _,row,error,angle=min(candidates,key=lambda a:a[0]);points.append(row)
            details.append(dict(fraction=float(t),position_mm=float(error),orientation_deg=float(angle)))
    checkpoint(p)
    positions=np.array(points)
    result.optimized_plan=JointState.from_numpy(names,positions,tensor_args=p.tensor_args)
    result.x_refinement=dict(source='GPU_IK_local_knots_after_MotionGen',knots=len(points),
        seconds=time.monotonic()-began,checks=details,execution_segments=1)
    return result
