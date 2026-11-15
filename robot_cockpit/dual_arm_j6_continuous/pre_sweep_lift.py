"""Measured-start +Z lift before J6/J7 alignment; millimetres externally."""
import numpy as np
STAGE='j6_pre_sweep_lift'
from continuous_policy import MAX_PHASE_MM

def height_for_count(count):
    if type(count) is not int or count<1 or count+1>MAX_PHASE_MM/10:
        raise ValueError('书堆总本数无效或动态上抬超过连续动作范围')
    return float((count+1)*10)

def from_left_wrist(reference,batch_id,epoch):
    if (not batch_id or reference.get('batch_id')!=batch_id or reference.get('epoch')!=epoch
            or not reference.get('frame_token')):
        raise ValueError('书本计数不属于当前帧/任务/控制会话')
    count=reference.get('stack_book_count')
    if reference.get('stack_count_status')!='confirmed':
        raise ValueError('左腕GPT未确认书堆总本数，未生成动态上抬目标')
    return dict(lift_mm=height_for_count(count),stack_book_count=count,
                lift_source='initial_left_wrist_stack_count',frame_token=reference['frame_token'],
                batch_id=batch_id,epoch=epoch,gpt_count_enabled=True,formula='(N+1)*10mm')

def stage_lift_mm(stage):
    value=stage.get('lift_mm')
    expected=height_for_count(stage.get('stack_book_count'))
    if (stage.get('lift_source')!='initial_left_wrist_stack_count' or not stage.get('frame_token')
            or not stage.get('batch_id') or type(value) not in (int,float)
            or not np.isfinite(value) or abs(value-expected)>1e-6):
        raise ValueError('动态上抬记录与本轮书本计数不一致')
    return expected

def evidence_for(state,joints,grip,xyz_m,wxyz,*,lift_config):
    q=np.asarray(joints,float);g=np.asarray(grip,float)
    p=np.asarray(xyz_m,float);r=np.asarray(wxyz,float)
    if q.shape!=(14,) or g.shape!=(2,) or p.shape!=(3,) or r.shape!=(4,) or not np.isfinite(np.r_[q,g,p,r]).all():
        raise ValueError('抬升起点反馈无效')
    distance=stage_lift_mm(lift_config)
    if state['epoch']!=lift_config.get('epoch'):
        raise ValueError('动态上抬控制会话已改变')
    target=p.copy();target[2]+=distance*.001
    return dict(epoch=state['epoch'],start_joints_rad=q.tolist(),gripper_feedback=g.tolist(),
                right_tcp_mm=(p*1000).tolist(),right_tcp_wxyz=r.tolist(),
                stages=[dict(stage=STAGE,translation_axis='Z',tcp_m_wxyz=np.r_[target,r].tolist(),
                             **{key:lift_config[key] for key in ('lift_mm','lift_source','stack_book_count','frame_token','batch_id')})])
