"""J6-specific continuation; old J7 entry points remain unchanged."""
import numpy as np


def run(source, model, token, report, preview, out):
    from right_approach_motion import execute
    from right_plane_regrasp import run as regrasp
    if report.get('rotate_j6_only', {}).get('status') != 'confirmed':
        raise RuntimeError('J6下拨未确认，不回收或抓取')
    for name in ('withdraw_x', 'raise_z50', 'withdraw_to_reset_x'):
        current = source.snapshot(token)
        joints, _, grip = source.state_arrays(current)
        xyz, quat = model.tcp(joints)
        initial = xyz[0,1]*1000
        target = initial.copy()
        stage = dict(stage=name, translation_axis='X')
        if name == 'withdraw_x':
            target[0] -= 60.
            print('J6下拨已确认；沿基座-X回收60mm，保持当前Y/Z和姿态。', flush=True)
        elif name == 'raise_z50':
            target[2] += 50.
            stage['translation_axis'] = 'Z'
            print('回收已确认；额外沿+Z抬升50mm，保持当前X/Y和姿态。', flush=True)
        else:
            reset_x = float(report['first_reset']['actual_tcp_mm'][0])
            if not np.isfinite(reset_x) or reset_x > initial[0]+3.:
                raise ValueError('本轮first_reset的X无效或不在回收方向；不继续')
            target[0] = reset_x
            stage['first_reset_x_mm'] = reset_x
        stage['tcp_m_wxyz'] = np.r_[target*.001,quat[0,1]].tolist()
        evidence = dict(epoch=current['epoch'],start_joints_rad=joints.tolist(),
                        gripper_feedback=grip.tolist(),right_tcp_mm=initial.tolist(),
                        right_tcp_wxyz=quat[0,1].tolist(),stages=[stage])
        record = report[name] = dict(evidence=evidence,status='requested')
        source.save_report(out,report)
        record['arrival'] = execute(evidence,token,source.state_arrays)
        record['status'] = 'confirmed'
        source.save_report(out,report)
    print('回收/抬升完成；衔接原J7后续：复位、J7顺90°、恢复Y、按首次左腕Z含补偿下降、前伸150mm闭爪及抓后动作。', flush=True)
    regrasp(source,model,token,report,preview,out,reset_and_restore_y=True,visual_alignment=False)
    report['status'] = 'j6_push_grasp_and_post_grasp_completed'
