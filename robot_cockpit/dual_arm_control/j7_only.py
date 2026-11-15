"""Independent trial J7 joint targets and pre-publication path guard."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ACTION='right_trial_j7_only'
MARKER=Path(f'/tmp/dual-arm-control-j7-only-{os.getuid()}.json')
HOLD_RAD=0.002  # numerical/feedback tolerance, not a commanded extra movement


def revision():
    root=Path(__file__).parent
    sources = [root/p for p in ('j7_only.py', 'planning_diagnostics.py', 'j7_trajectory.py', 'planning_receipt.py')]
    sources.append(root.parents[1]/'dual_v2_2_control/cuarm_upper_planner/scripts/book_lower_pose.py')
    sources.append(root.parents[1]/'dual_v2_2_control/cuarm_upper_planner/scripts/curobo_planner_v3_1.py')
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()


def publish_capability():
    MARKER.write_text(json.dumps(dict(pid=os.getpid(),revision=revision())))


def require_capability(pid):
    try:d=json.loads(MARKER.read_text())
    except (OSError,ValueError):d={}
    if pid is None or d.get('pid')!=pid or d.get('revision')!=revision():
        raise RuntimeError('需要匹配的新版试验Planner（J7检查/速度/Z姿态求解）：先正常停止控制，再运行 bash /mnt/ssd/user/V/dual_arm_start.sh --restart-planner')


def joint_goal(model,joints, *, clockwise=True):
    q=np.asarray(joints,float).copy()
    _,rotation=model.tcp(q)
    probe=q.copy();probe[13]+=.001
    _,r2=model.tcp(probe)
    r=Rotation.from_quat(rotation[0,1,[1,2,3,0]])
    derivative=(r.inv()*Rotation.from_quat(r2[0,1,[1,2,3,0]])).as_rotvec()/.001
    tool=np.array(model.offsets[1],dtype=float,copy=True);tool/=np.linalg.norm(tool)
    projection=float(derivative@tool)
    if not np.isfinite(projection) or abs(projection)<.5:
        raise ValueError('J7轴无法对应此前顺时针工具方向；未发送运动')
    delta=90. if projection>0 else -90.
    if not clockwise:
        delta = -delta
    q[13]+=np.deg2rad(delta)
    if not model.bounds[0,13]<=q[13]<=model.bounds[1,13]:
        raise ValueError('仅J7旋转90度超过关节限位；不改用其他关节补偿')
    return q,delta


def check_path(samples,start,names):
    samples=np.asarray(samples,float).reshape(-1,len(names));start=np.asarray(start,float).reshape(-1)
    if samples.size==0 or start.shape!=(len(names),) or not np.isfinite(samples).all():
        raise ValueError('J7 trajectory data invalid')
    moving=names.index('arm_J7_r');fixed=[i for i in range(len(names)) if i!=moving]
    error=float(np.max(np.abs(samples[:,fixed]-start[fixed])))
    if error>HOLD_RAD:
        raise ValueError(f'仅J7轨迹仍带动其他关节：max={np.rad2deg(error):.4f}deg；拒绝下发')
    return error
