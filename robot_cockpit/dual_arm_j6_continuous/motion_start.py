"""Bind a bounded, settled execution start without moving absolute task targets.

Caller must own the same authorized session and validate fresh paired feedback.
"""
import copy
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from approach_handoff import JOINT_START_LIMIT_RAD

from pre_sweep_lift import STAGE, stage_lift_mm

POSITION_LIMIT_MM = 3.0
ORIENTATION_LIMIT_DEG = 0.5
RELATIVE = {'withdraw_x': ('X', -60.), 'lower_z5': ('Z', -5.), 'raise_z50': ('Z', 50.),
            'post_grasp_raise_z200': ('Z', 200.), 'post_grasp_lower_z200': ('Z', -200.)}
ABSOLUTE = {'approach_y': 'Y', 'advance_x': 'X', 'withdraw_to_reset_x': 'X', 'cached_top_lower_z': 'Z'}


def bind_settled_start(evidence, joints, xyz_mm, wxyz):
    old = copy.deepcopy(evidence['stages'][0])
    name = old['stage']
    relative=dict(RELATIVE)
    if name==STAGE:relative[name]=('Z',stage_lift_mm(old))
    q, p, r = np.asarray(joints,float), np.asarray(xyz_mm,float), np.asarray(wxyz,float)
    oq=np.asarray(evidence['start_joints_rad'],float)
    op=np.asarray(evidence['right_tcp_mm'],float);orr=np.asarray(evidence['right_tcp_wxyz'],float)
    if (q.shape!=(14,) or p.shape!=(3,) or r.shape!=(4,) or oq.shape!=(14,) or op.shape!=(3,)
            or orr.shape!=(4,) or not np.isfinite(np.r_[q,p,r,oq,op,orr]).all()):
        raise ValueError('执行起点反馈维度或数值无效')
    angle=float(np.rad2deg((Rotation.from_quat(orr[[1,2,3,0]]).inv()*Rotation.from_quat(r[[1,2,3,0]])).magnitude()))
    delta=p-op
    record=dict(unix_s=time.time(),stage=name,original_tcp_mm=op.tolist(),execution_tcp_mm=p.tolist(),
                original_wxyz=orr.tolist(),execution_wxyz=r.tolist(),tcp_delta_mm=delta.tolist(),
                original_joints_rad=oq.tolist(),execution_joints_rad=q.tolist(),
                position_error_mm=float(np.linalg.norm(delta)),orientation_error_deg=angle,
                max_joint_delta_rad=float(np.max(np.abs(q-oq))),position_limit_mm=POSITION_LIMIT_MM,
                orientation_limit_deg=ORIENTATION_LIMIT_DEG,joint_limit_rad=JOINT_START_LIMIT_RAD,
                original_stage=old,accepted=False)
    evidence['execution_start_binding']=record
    if (record['position_error_mm']>POSITION_LIMIT_MM or angle>ORIENTATION_LIMIT_DEG
            or record['max_joint_delta_rad']>JOINT_START_LIMIT_RAD):
        raise RuntimeError('停稳后的执行起点偏移超出允许范围；'+json.dumps(record,ensure_ascii=False))
    if name not in relative and name not in ABSOLUTE:
        # Joint rotations and legacy visual servo retain their original semantics.
        if name not in ('rotate_j6_only', 'rotate_j7_fingers_y', 'rotate_j7_only', 'j6_axis_align') and record['position_error_mm']>1.:
            raise RuntimeError('此动作不支持起点重算；'+json.dumps(record,ensure_ascii=False))
        record['policy']='original_target_unchanged'
        record['accepted']=True
        return
    axis=relative.get(name, (ABSOLUTE.get(name),))[0]
    if old.get('translation_axis','Y')!=axis or old.get('hold_pose_mm_wxyz') is not None:
        raise ValueError('动作轴或固定视觉参考与起点绑定不一致')
    idx='XYZ'.index(axis);fixed=[i for i in range(3) if i!=idx]
    target=np.asarray(old['tcp_m_wxyz'],float)
    if target.shape!=(7,) or not np.isfinite(target).all() or not np.allclose(target[fixed]*1000,op[fixed],atol=1e-6,rtol=0):
        raise ValueError('原动作不是有效的单轴目标；不重算')
    new=copy.deepcopy(old); endpoint=p.copy()
    if name in relative:
        distance=relative[name][1]
        if not np.isclose(target[idx]*1000-op[idx],distance,atol=1e-6,rtol=0):
            raise ValueError('原相对位移与动作定义不一致')
        endpoint[idx]+=distance
        record['policy']='preserve_relative_distance'
    elif name=='advance_x' and old.get('close_right_gripper_after_arrival'):
        if not np.isclose(target[0]*1000-op[0],150.,atol=1e-6,rtol=0):
            raise ValueError('抓取前伸必须保持150mm')
        endpoint[0]+=150.
        record['policy']='preserve_grasp_150mm'
    else:
        endpoint[idx]=target[idx]*1000
        record['policy']='preserve_absolute_axis_goal'
    new['tcp_m_wxyz']=np.r_[endpoint*.001,r].tolist()
    record.update(accepted=True,execution_stage=new)
    evidence['stages'][0]=new
    evidence.update(start_joints_rad=q.tolist(),right_tcp_mm=p.tolist(),right_tcp_wxyz=r.tolist())
    print('停稳后绑定执行起点：'+json.dumps(record,ensure_ascii=False),flush=True)
