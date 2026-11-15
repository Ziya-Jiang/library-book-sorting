"""Scoped raw IK constraint evidence. Read-only metrics, no bounds relaxed."""
from contextlib import contextmanager
import numpy as np

@contextmanager
def capture(solver,arm_names,start,report):
    original=solver._get_result
    def inspect(num_seeds,result,goal_pose,return_seeds):
        try:
            safety=solver.solver.safety_rollout
            action=result.action if result.action.position.ndim==3 else result.action.unsqueeze(1)
            state=safety.compute_kinematics(action)
            terms={'bounds':safety.bound_constraint.forward(state.state_seq)}
            for label,attr in [('self_collision','robot_self_collision_constraint'),('world_collision','primitive_collision_constraint')]:
                term=getattr(safety,attr,None)
                if term is not None and term.enabled:terms[label]=term.forward(state.robot_spheres)
            cpu=lambda v:v.detach().cpu().numpy().copy()
            data={k:cpu(v).reshape(-1) for k,v in terms.items()}
            for k in ('feasible','constraint','position_error','rotation_error'):
                data[k]=cpu(getattr(result.metrics,k)).reshape(-1)
            rows=cpu(result.action.position).reshape(-1,len(solver.joint_names))
            limits=cpu(safety.bound_constraint.joint_limits.position)
            out=[]
            for i,row in enumerate(rows):
                info={k:float(v[i]) for k,v in data.items()}
                info['bound_violations_deg']={n:float(np.rad2deg(max(limits[0,j]-row[j],row[j]-limits[1,j],0.)))
                    for j,n in enumerate(solver.joint_names) if row[j]<limits[0,j] or row[j]>limits[1,j]}
                info['right_joint_delta_deg']=[float(np.rad2deg(row[solver.joint_names.index(n)]-start[j]))
                                              for j,n in enumerate(arm_names) if j>=7]
                out.append(info)
            report.setdefault('raw_ik_calls',[]).append(out)
            all_rows=[row for call in report['raw_ik_calls'] for row in call]
            best=min(all_rows,key=lambda row:row['constraint']+row['position_error']+row['rotation_error'])
            report['best_ik_candidate']=dict(position_error_mm=best['position_error']*1000.,
                rotation_error_native=best['rotation_error'],bound_violations_deg=best['bound_violations_deg'],
                right_joint_delta_deg=best['right_joint_delta_deg'],self_collision=best.get('self_collision'),
                world_collision=best.get('world_collision'))
            report['position_threshold_m']=float(solver.position_threshold)
            report['rotation_threshold_native']=float(solver.rotation_threshold)
        except Exception as error:
            report['diagnostic_error']=type(error).__name__+': '+str(error)
        return original(num_seeds,result,goal_pose,return_seeds)
    solver._get_result=inspect
    try:yield
    finally:solver._get_result=original
