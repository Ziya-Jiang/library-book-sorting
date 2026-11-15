"""Right-wrist top-spine alignment after J7 rotation; no cached height target."""
import json
import time
import numpy as np
from right_regrasp import WristWorker, select_spine


def align_axis(axis, observe, move, coordinate, report):
    """Learn local pixel response from measured movement. Z commands only descend."""
    if axis!='Z':raise ValueError('This grasp stage only permits Z alignment')
    tolerance=3.;start=coordinate();began=time.monotonic();stable=0;slopes=[]
    error=observe(axis)
    if abs(error)>tolerance:
        # Two same-direction probes avoid an upward Z return probe.
        for _ in range(2):
            old=coordinate();before=error
            move(axis,-1. if axis=='Z' else 1.)
            current=coordinate();error=observe(axis)
            delta=current-old
            if abs(delta)<.3:raise ValueError('腕部伺服探测实测位移不足')
            slopes.append((error-before)/delta)
            if abs(error)<=tolerance:break
        if min(abs(s) for s in slopes)<.2 or (len(slopes)==2 and slopes[0]*slopes[1]<=0):
            raise ValueError('腕部图像轴响应不可观测或方向不一致')
    jacobian=float(np.mean(slopes)) if slopes else None
    report.update(jacobian_px_per_mm=jacobian,probe_slopes=slopes)
    for _ in range(100):
        if time.monotonic()-began>120:raise ValueError('腕部视觉对齐超过120秒')
        error=observe(axis)
        if abs(error)<=tolerance:
            stable+=1
            if stable>=3:
                report.update(aligned=True,final_error_px=error,elapsed_s=time.monotonic()-began)
                return
            continue
        stable=0
        if jacobian is None:raise ValueError('中线离开容差且无有效图像轴响应')
        delta=float(np.clip(-.6*error/jacobian,-3.,3.))
        if axis=='Z' and delta>=0:raise ValueError('中线对齐需要向上修正；按仅-Z要求停止，不前伸')
        if abs(coordinate()+delta-start)>(80. if axis=='Z' else 180.):
            raise ValueError('腕部视觉对齐超过本轮行程边界')
        move(axis,delta)
    raise ValueError('腕部视觉对齐未收敛，不前伸')


def run(source,model,token,previous_report,preview,out):
    from right_approach_motion import execute
    from approach_handoff import JOINT_START_LIMIT_RAD
    report=dict(status='started',head_recaptured=False,cached_z_used=True,motions=[],observations=[])
    previous_report['wrist_visual_grasp']=report
    worker=None
    def state():
        s=source.snapshot(token);j,_,g=source.state_arrays(s);p,q=model.tcp(j)
        return s,j,g,p[0,1]*1000,q[0,1]
    def move(axis,delta):
        s,j,g,p,q=state();target=p.copy();i='XYZ'.index(axis)
        target[i]+=delta
        fixed=[k for k in range(3) if k!=i];target[fixed]=hold[:3][fixed]
        stage=dict(stage='wrist_'+axis.lower()+'_servo',translation_axis=axis,
                   hold_pose_mm_wxyz=hold.tolist(),tcp_m_wxyz=np.r_[target*.001,hold[3:]].tolist(),
                   downward_only=axis=='Z')
        evidence=dict(epoch=s['epoch'],start_joints_rad=j.tolist(),gripper_feedback=g.tolist(),
                      right_tcp_mm=p.tolist(),right_tcp_wxyz=q.tolist(),stages=[stage])
        record=dict(status='requested',evidence=evidence);report['motions'].append(record)
        record['arrival']=execute(evidence,token,source.state_arrays);record['status']='confirmed'
    try:
        worker=WristWorker();before=state();capture=worker.request('capture')
        report.update(selection_frame=capture['frame_token'],worker_release=worker.release)
        print('复位、旋转、Y Approach及含补偿Z移动完成：等待GPT-6.1分割右腕最上层书脊，再按中线-Z伺服。',flush=True)
        selection=select_spine(capture,source.api_key());after=state()
        if (before[0]['epoch']!=after[0]['epoch'] or
            np.max(np.abs(before[1]-after[1]))>JOINT_START_LIMIT_RAD or
            not np.allclose(before[2],after[2],atol=1e-6,rtol=0)):
            raise ValueError('右腕GPT等待期间姿态/夹爪/会话改变；不开始伺服')
        report['selection']=selection
        worker.request('init',frame_token=capture['frame_token'],corners_px=selection['corners'])
        def observe(axis):
            observation=worker.request('observe',axis=axis)
            report['observations'].append({k:v for k,v in observation.items() if k!='jpeg'})
            if preview is not None:preview.show_right_wrist(observation)
            key='error_u_px' if axis=='Y' else 'error_v_px'
            error=float(observation[key])
            if not np.isfinite(error):raise ValueError('书脊中线测量无效')
            print(f'右腕{axis}对齐：中线误差={error:.2f}px；点数={len(observation["point_ids"])}',flush=True)
            return error
        for axis in ('Z',):
            sample=state();hold=np.r_[sample[3],sample[4]]
            section=report.setdefault(axis+'_alignment',{})
            align_axis(axis,observe,move,lambda:float(state()[3]['XYZ'.index(axis)]),section)
        # Confirm both features again before any forward movement.
        final=worker.request('observe',axis='Z')
        if preview is not None:preview.show_right_wrist(final)
        report['final_observation']={k:v for k,v in final.items() if k!='jpeg'}
        if abs(final['error_v_px'])>3:
            raise ValueError('前伸前Z中线复核未通过，不前伸')
        s,j,g,p,q=state();target=p.copy();target[0]+=100.
        evidence=dict(epoch=s['epoch'],start_joints_rad=j.tolist(),gripper_feedback=g.tolist(),
                      right_tcp_mm=p.tolist(),right_tcp_wxyz=q.tolist(),stages=[dict(stage='advance_x',
                      translation_axis='X',tcp_m_wxyz=np.r_[target*.001,q].tolist(),close_right_gripper_after_arrival=True)])
        report['forward']=dict(evidence=evidence,status='requested')
        report['forward']['arrival']=execute(evidence,token,source.state_arrays)
        report['forward']['status']='confirmed';report['status']='visual_aligned_forward_and_close_echoed'
        print('右腕书脊中线Z对齐已确认，前伸100mm完成且闭爪指令已回显；保持当前位置。',flush=True)
    except Exception as error:
        report.update(status='stopped',reason=str(error));raise
    finally:
        if worker is not None:worker.close()
        (out/'wrist_visual_grasp.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
