"""Offline ready-pose ranking and fail-closed candidate loading.

Incomplete historical cases never establish an improvement. All comparisons
use the same model/world contract and fixed measured left-arm configuration.
"""
import hashlib
import json
import dataclasses
from pathlib import Path
import numpy as np
from baseline_approach_path import Search, seeds, Rejected, angle

CONFIG=Path(__file__).resolve().parents[3]/'dual_v2_2_agent/baseline_ready_pose.json'


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
    if hasattr(value,'detach'):return value.detach().cpu().tolist()
    if type(value).__module__=='torch' and type(value).__name__ in ('device','dtype'):return str(value)
    raise TypeError('Unsupported model contract value: '+type(value).__name__)


def contract(planner):
    p=planner;b=p._v31_bridge
    k=p._motion_gen.kinematics.kinematics_config
    data=dict(limits=np.asarray(p.arm_soft_joint_limits).tolist(),offsets=np.asarray(p.end_effector_offset).tolist(),
              links=list(p.target_link_names),names=list(p.total_arm_joint_names),world=p.world_cfg,
              spheres=k.link_spheres,sphere_links=k.link_sphere_idx_map,
              self_collision=p._feedback_kinematics.get_self_collision_config(),
              dynamics=k.joint_limits,
              fk={name:np.asarray(getattr(b.cpu,name)).tolist() for name in
                  ('fixed','parent','joint','kind','offset','store','dof')})
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False,default=json_value).encode()).hexdigest()


def candidate(planner, original, path=CONFIG):
    data=json.loads(Path(path).read_text())
    if data.get('enabled') is not True:
        return None
    if data.get('protocol')!=1 or data.get('model_contract')!=contract(planner):
        raise Rejected('Ready candidate model/world contract mismatch')
    score=data.get('validation',{})
    if (score.get('complete_cases',0)<1 or score.get('incomplete_cases',1)!=0
            or score.get('candidate_score',[])<=score.get('original_score',[])):
        raise Rejected('Ready candidate lacks complete comparative replay evidence')
    if not np.allclose(np.deg2rad(data['reference_right_deg']),original[7:],atol=1e-8,rtol=0):
        raise Rejected('Ready reference changed (including configured J6)')
    if not np.allclose(data['left_joint_rad'],original[:7],atol=.002,rtol=0):
        raise Rejected('Ready candidate left-arm collision context changed')
    result=original.copy();result[7:]=np.deg2rad(data['candidate_right_deg'])
    if not np.isfinite(result).all():
        raise Rejected('Nonfinite ready candidate')
    oldp,oldq=planner._v31_bridge.tcp(original);newp,newq=planner._v31_bridge.tcp(result)
    if np.linalg.norm(oldp[0,1]-newp[0,1])>.001 or angle(oldq[0,1],newq[0,1])>.25:
        raise Rejected('Ready candidate changed original TCP')
    bounds=np.asarray(planner.arm_soft_joint_limits)
    if np.min(np.minimum(result[7:]-bounds[0,7:],bounds[1,7:]-result[7:]))<np.deg2rad(1):
        raise Rejected('Ready candidate lacks joint reserve')
    return result


def optimize(adapter_factory, original, cases, evaluate_case, model_contract):
    """Offline API. evaluate_case(q,case) must replay Y/roll alignment too.

    It returns a validated advance path's metrics. Merely testing original
    alignment joints would not evaluate a changed ready pose and is forbidden.
    """
    required={'start_joint_rad','goal_xyz_m','goal_wxyz','model_contract',
              'alignment_history','world_snapshot','gripper_state'}
    incomplete=[i for i,c in enumerate(cases) if not required.issubset(c) or c['model_contract']!=model_contract
                or not c['alignment_history']]
    base=dict(protocol=1,enabled=False,model_contract=model_contract,
              reference_right_deg=np.rad2deg(original[7:]).tolist(),left_joint_rad=original[:7].tolist(),
              validation=dict(complete_cases=len(cases)-len(incomplete),incomplete_cases=len(incomplete)))
    if incomplete or not cases:
        return dict(base,reason='Incomplete historical replay; original ready pose retained')
    a=adapter_factory();pos,quat=a.tcp(original)
    candidates=[original.copy()]
    base['candidate_rejections']=[]
    for index,seed in enumerate(seeds(original,a.bounds)):
        search=Search(adapter_factory())
        try:
            q=search.project(seed,pos[0,1],quat[0,1])
            search.a.collision(q[None],search.check)
            if not any(np.max(np.abs(q-x))<.002 for x in candidates):candidates.append(q)
        except (Rejected,TimeoutError) as error:
            base['candidate_rejections'].append(dict(seed=index,reason=str(error)))
    ranked=[]
    for index,q in enumerate(candidates):
        results=[]
        for case_index,case in enumerate(cases):
            try:
                r=evaluate_case(q,case)
                if r.get('replayed_alignment') is not True or r.get('validated_rt_curve') is not True:
                    raise Rejected('Incomplete alignment/path replay')
                results.append(r)
            except (Rejected,TimeoutError) as error:
                base['candidate_rejections'].append(dict(candidate=index,case=case_index,reason=str(error)))
        score=(len(results),min((r['min_path_margin_deg'] for r in results),default=-1.),
               -float(np.linalg.norm(q-original)))
        ranked.append((score,q))
    old=ranked[0][0];best,q=max(ranked,key=lambda x:x[0])
    base['validation'].update(original_score=list(old),candidate_score=list(best))
    if best<=old:return dict(base,reason='No improved candidate; original ready pose retained')
    return dict(base,enabled=True,candidate_right_deg=np.rad2deg(q[7:]).tolist(),
                reason='Comparative replay passed; every live reset still requires a fresh collision-checked path')


def audit_events(paths):
    rows=[]
    for path in paths:
        events=[json.loads(line) for line in Path(path).read_text().splitlines()]
        stops=[r for r in events if r.get('event')=='board_control_stop']
        stop=stops[-1].get('stop_detail',{}) if stops else {}
        rows.append(dict(path=str(path),complete=False,replayed=False,
            missing=['coherent alignment joint snapshot','collision world/model contract',
                     'gripper state','replayable alignment history from alternative ready poses'],
            stop={key:stop.get(key) for key in ('reason','utc','rt_feedback_age_ms','target_id')}))
    return dict(cases=rows,complete_cases=0,selected_candidate=None,
                conclusion='Existing visual event logs alone cannot validate a new ready configuration')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read-only historical scene completeness audit; never enables a ready candidate')
    parser.add_argument('events',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit_events(args.events)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
