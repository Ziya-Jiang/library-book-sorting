"""Independent trial J6 joint targets and pre-publication path guard."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ACTION='right_trial_j6_only'
J7_ACTION='right_trial_j7_fingers_y'

def moving_joint(description):
    action=str(description).split(':',1)[0]
    return 'arm_J6_r' if action==ACTION else ('arm_J7_r' if action==J7_ACTION else ('arm_J7_r' if action=='right_trial_j7_only' else None))

def validate_result_before_publish(planner, result):
    """Match the publication validator to the generator used for this action."""
    action = str(planner.target_description).split(':',1)[0]
    if action == 'right_trial_j7_only':
        from j7_trajectory import validate_before_publish
        marker = 'single_j7_diagnostics'
    elif action in (ACTION, J7_ACTION):
        from j6_trajectory import validate_before_publish
        marker = 'single_j6_diagnostics'
    else:
        raise ValueError('Unknown single-wrist-joint action')
    if not hasattr(result, marker):
        raise ValueError(f'{action} missing matching trajectory proof: {marker}')
    validate_before_publish(planner, result)


MARKER=Path(f'/tmp/dual-arm-control-j6-only-{os.getuid()}.json')
HOLD_RAD=0.002  # numerical/feedback tolerance, not a commanded extra movement


def revision():
    root=Path(__file__).parent
    sources = [root/p for p in ('j6_only.py', 'planning_diagnostics.py', 'j6_trajectory.py', 'planning_receipt.py', 'j6_geometry.py', 'j7_only.py', 'j7_trajectory.py', 'wrist_timing.py', 'lift_continuity.py', 'gpu_preparation.py')]
    sources.append(root.parents[1]/'dual_v2_2_control/cuarm_upper_planner/scripts/book_lower_pose.py')
    sources.append(root.parents[1]/'dual_v2_2_control/cuarm_upper_planner/scripts/curobo_planner_v3_1.py')
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()


def publish_capability():
    MARKER.write_text(json.dumps(dict(pid=os.getpid(),revision=revision())))


def require_capability(pid):
    try:d=json.loads(MARKER.read_text())
    except (OSError,ValueError):d={}
    if pid is None or d.get('pid')!=pid or d.get('revision')!=revision():
        raise RuntimeError('需要匹配的新版试验Planner（J6检查/速度/Z姿态求解）：先正常停止控制，再运行 bash /mnt/ssd/user/V/right_arm_j6_start.sh --restart-planner')


def joint_goal(model, joints, grippers=None):
    from j6_geometry import downward_goal
    goal, detail = downward_goal(model,joints,grippers)
    return goal, detail['joint_delta_deg']


def check_path(samples,start,names,moving_name="arm_J6_r"):
    samples=np.asarray(samples,float).reshape(-1,len(names));start=np.asarray(start,float).reshape(-1)
    if samples.size==0 or start.shape!=(len(names),) or not np.isfinite(samples).all():
        raise ValueError('J6 trajectory data invalid')
    moving=names.index(moving_name);fixed=[i for i in range(len(names)) if i!=moving]
    error=float(np.max(np.abs(samples[:,fixed]-start[fixed])))
    if error>HOLD_RAD:
        raise ValueError(f'仅J6轨迹仍带动其他关节：max={np.rad2deg(error):.4f}deg；拒绝下发')
    return error
