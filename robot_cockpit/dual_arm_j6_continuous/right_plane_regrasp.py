"""Reuse initial left-wrist spine height; optional reset/Y restore, no camera or model API."""
import json
import time
import numpy as np


def run(source, model, token, previous_report, preview, out, *, reset_and_restore_y=False, visual_alignment=False):
    from right_approach_motion import execute
    report = dict(status='started', second_gpt_called=False, head_recaptured=False,
                  first_reset_repeated=False, motions=[])
    previous_report['post_withdraw_regrasp'] = report

    def state():
        s = source.snapshot(token)
        j, _, g = source.state_arrays(s)
        p, q = model.tcp(j)
        return s, j, g, p[0, 1]*1000, q[0, 1]

    def move(name, axis, target, s, j, g, p, q, **fields):
        stage = dict(stage=name, translation_axis=axis,
                     tcp_m_wxyz=np.r_[target*.001, q].tolist(), **fields)
        evidence = dict(epoch=s['epoch'], start_joints_rad=j.tolist(), gripper_feedback=g.tolist(),
                        right_tcp_mm=p.tolist(), right_tcp_wxyz=q.tolist(), stages=[stage])
        record = dict(evidence=evidence, status='requested')
        report['motions'].append(record)
        record['arrival'] = execute(evidence, token, source.state_arrays)
        record['status'] = 'confirmed'
        return record['arrival']

    try:
        surface = previous_report['first_left_spine']
        if not surface.get('batch_id') or surface.get('batch_id') != previous_report.get('perception_batch_id'):
            raise ValueError('左腕书脊Z不属于本轮初始感知；未发送运动')
        z = float(surface['target_z_mm'])
        if not np.isfinite(z):
            raise ValueError('首次左腕书脊Z无效；未发送运动')
        s, j, g, p, q = state()
        if reset_and_restore_y:
            from preparation_target import grasp_y_after_push
            saved_y = grasp_y_after_push(previous_report,p[1])
            reset_goal = np.asarray(previous_report['first_reset']['target_right_joint_deg'], float)
            if reset_goal.shape != (7,) or not np.isfinite(reset_goal).all() or not np.isfinite(saved_y):
                raise ValueError('本轮复位目标或回拨Y无效；未发送复位')
            reset = dict(epoch=s['epoch'], start_joints_rad=j.tolist(), gripper_feedback=g.tolist(),
                         target_right_joint_deg=reset_goal.tolist())
            record = dict(stage='first_reset', evidence=reset, status='requested')
            report.update(saved_y_mm=saved_y, before_reset_tcp_mm=p.tolist())
            report['motions'].append(record)
            print(f'回拨后先松右爪再完整first_reset；抓取对准Y={saved_y:.3f}mm。', flush=True)
            record['arrival'] = execute(reset, token, source.state_arrays, first_reset=True)
            record['status'] = 'confirmed'
            report['first_reset_repeated'] = True
            s, j, g, p, q = state()
            report['reset_tcp_mm'] = p.tolist()
            from j7_only import joint_goal
            joint_target, delta = joint_goal(model, j)
            rotation = dict(epoch=s['epoch'], start_joints_rad=j.tolist(), gripper_feedback=g.tolist(),
                            right_tcp_mm=p.tolist(), right_tcp_wxyz=q.tolist(),
                            stages=[dict(stage='rotate_j7_only', target_joints_rad=joint_target.tolist(),
                                         j7_delta_deg=delta, tcp_m_wxyz=np.r_[p*.001,q].tolist())])
            record = dict(stage='rotate_j7_only', evidence=rotation, status='requested')
            report['motions'].append(record)
            print(f'回拨后first_reset已确认；仅右J7 {delta:+.1f}°（顺时针90°），到位后恢复Y。', flush=True)
            record['arrival'] = execute(rotation, token, source.state_arrays)
            record['status'] = 'confirmed'
            s, j, g, p, q = state()
            offset_y = saved_y - float(p[1])
            report.update(after_reset_rotation_tcp_mm=p.tolist(), restore_y_offset_mm=offset_y)
            target = p.copy(); target[1] = saved_y
            print(f'恢复抓取书脊中心Y：旋转后Y={p[1]:.3f}mm，目标Y={saved_y:.3f}mm，offset_Y={offset_y:+.3f}mm；保持旋转后的X/Z和姿态。', flush=True)
            move('approach_y', 'Y', target, s, j, g, p, q)
            s, j, g, p, q = state()
        offset = z - float(p[2])
        report.update(reference=surface, target_z_mm=z, offset_z_mm=offset,
                      initial_tcp_mm=p.tolist(), reference_age_s=time.time()-surface['captured_unix_s'])
        if offset > 0:
            raise ValueError(f'首次左腕书脊Z高于当前TCP {offset:.3f}mm，无法按−Z下降；未发送运动')
        target = p.copy(); target[2] = z
        raw_z = float(surface.get('measured_spine_z_mm', z))
        compensation = float(surface.get('base_z_compensation_mm', 0.))
        message = (f'首次左腕书脊Z={raw_z:.3f}mm；Z补偿={compensation:.1f}mm；目标Z={z:.3f}mm；'
                   f'当前Z={p[2]:.3f}mm，offset_Z={offset:.3f}mm；固定X/Y和姿态下降。')
        print(message, flush=True)
        if preview is not None:
            preview.status(message)
            show_offset = getattr(preview, 'show_z_offset', None)
            if callable(show_offset):
                show_offset(float(p[2]), planned_offset=offset)
            else:
                report['preview_warning'] = '旧预览对象无show_z_offset；Z/offset已显示在任务状态中'
                print(report['preview_warning'], flush=True)
        move('cached_top_lower_z', 'Z', target, s, j, g, p, q, target_top_z_mm=z)
        if visual_alignment:
            from wrist_visual_alignment import run as align_and_grasp
            report.update(second_gpt_called=True,cached_z_used=True)
            align_and_grasp(source,model,token,previous_report,preview,out)
            report['status']='wrist_visual_grasp_completed'
            return
        s, j, g, p, q = state()
        target = p.copy(); target[0] += 150.
        report.update(forward_start_tcp_mm=p.tolist(), forward_offset_mm=150., target_x_mm=float(target[0]))
        arrival = move('advance_x', 'X', target, s, j, g, p, q, close_right_gripper_after_arrival=True)
        report['gripper'] = arrival['gripper']
        report['status'] = 'forward_arrived_and_close_command_echoed'
        previous_report['post_grasp_motion'] = dict(
            status='disabled', reason='hold_after_grasp_requested', motions=[])
        print('下降及前伸已到位，右夹爪闭合指令已回显；保持当前位置，抓取后不再抬升、旋转或下降。夹爪回显不代表已确认抓稳书。', flush=True)
    except Exception as error:
        report.update(status='stopped', reason=str(error))
        raise
    finally:
        (out/'post_withdraw_regrasp.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
