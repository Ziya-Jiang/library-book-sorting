"""Continuous J6: left-wrist RGB-D task perception; optional head live preview."""

import argparse
from datetime import datetime
import getpass
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np

import start
from preview_runtime import choose_output, save_report, OptionalPreview
from backend.board_servo.model import Model

AGENT = start.V / 'dual_v2_2_agent'
sys.path[:0] = [str(start.V), str(start.V / 'robot_cockpit/launcher'), str(AGENT)]

from head_cover_y import wrist_spine_y


def api_key():
    key = os.environ.pop('OPENAI_API_KEY', '').strip()
    alias = os.environ.pop('OPENAI_KEY', '').strip()
    key = key or alias
    credential_file = start.V / '.secrets/right_approach_openai.key'
    if not key and credential_file.is_file():
        key = credential_file.read_text(encoding='utf-8').strip()
    if not key:
        if not sys.stdin.isatty():
            raise ValueError('请在100本机设置 OPENAI_API_KEY 或在交互终端输入 Key')
        key = getpass.getpass('OpenAI API Key（100本机，不回显）: ').strip()
    if not key or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError('OpenAI API Key 格式无效')
    return key


def verified_model():
    config = start.ROOT / 'configuration/dual_gripper.urdf'
    bundle = start.ROOT / 'backend/board_servo'
    local = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, bundle/'manifest.json')]
    remote = start.remote(['sha256sum', start.REMOTE + '/configuration/dual_gripper.urdf',
                           start.REMOTE + '/backend/board_servo/manifest.json'], timeout=8)
    if [line.split()[0] for line in remote.splitlines()] != local:
        raise ValueError('100的右臂模型与155已部署模型不一致')
    model = Model(bundle)
    if model.contract.get('model') != local[0]:
        raise ValueError('CuRobo URDF 与运动学导出不一致')
    return model, local[0]


def snapshot(token):
    request = Request(start.GUI + '/api/state', headers={'X-Arm-Token': token})
    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
        return json.load(response)


def state_arrays(state):
    if (state.get('rt_assembly') != 'dual_gripper' or state.get('simulation') is not False
            or state.get('gripper_installed') != [True, True] or not state.get('rt_running')
            or not state.get('fresh') or not 0 <= state['age_ms'] <= 100):
        raise ValueError('需要155双夹爪配置的新鲜RT反馈')
    arms = state['state']['arms']
    joints = np.asarray([arm['position'] for arm in arms], float).reshape(14)
    velocity = np.asarray([arm['velocity'] for arm in arms], float).reshape(14)
    grip = np.asarray([g['position'] for g in state['state']['grippers']], float).reshape(2)
    if not np.isfinite(np.r_[joints, velocity, grip]).all() or np.any(grip < 0) or np.any(grip > 1):
        raise ValueError('双臂或夹爪反馈无效')
    return joints, velocity, grip


def measure_approach(before, after, model, urdf_sha, report):
    old, initial_velocity, old_grip = state_arrays(before)
    joints, velocity, grip = state_arrays(after)
    if (before['epoch'] != after['epoch'] or np.max(np.abs(joints - old)) > np.deg2rad(.2)
            or np.max(np.abs(np.r_[initial_velocity, velocity])) > .02 or not np.array_equal(grip, old_grip)):
        raise ValueError('识别期间机械臂或夹爪状态改变，请从停稳位置重新采集')
    arms = after['state']['arms']
    soft = np.asarray([a['soft_position'] for a in arms], float).reshape(14, 2)
    hard = np.asarray([a['hard_position'] for a in arms], float).reshape(14, 2)
    bounds = np.stack([np.maximum.reduce([soft[:, 0], hard[:, 0], model.bounds[0]]),
                       np.minimum.reduce([soft[:, 1], hard[:, 1], model.bounds[1]])])
    speed = np.asarray([a['soft_velocity'] for a in arms], float).reshape(14, 2)
    caps = np.minimum(-speed[:, 0], speed[:, 1]) * .1
    pos, quat = model.tcp(joints)
    geometry = wrist_spine_y(report.get('first_left_spine', {}), report['perception_batch_id'],
                             after['epoch'], pos[0, 1]*1000, quat[0, 1])
    report['y_measurement_source'] = 'left_wrist_robot_facing_spine'
    report['depth'].update(sampling_region='left_wrist_robot_facing_spine',
                          source_frame_token=geometry['measured']['source_frame_token'],
                          sample_count=geometry['measured']['sample_count'])
    print(f'同轮左腕书脊实测Y={geometry["measured"]["center_mm"][1]:.3f}mm，'
          f'offset={geometry["offset_y_mm"]:+.3f}mm；头部仅预览。', flush=True)
    report.update(geometry, status='waypoints_ready', frame_token=geometry['measured']['source_frame_token'], epoch=after['epoch'],
                  board_model_sha256=model.digest, urdf_sha256=urdf_sha, start_joints_rad=joints.tolist(),
                  arm_limits_rad=bounds.tolist(), planner_velocity_caps_rad_s=caps.tolist(),
                  gripper_feedback=grip.tolist(), right_tcp_mm=(pos[0, 1] * 1000).tolist(),
                  right_tcp_wxyz=quat[0, 1].tolist())


def run(args, out, plan_out, report, preview=None):
    start.precheck()
    credential = api_key()
    url, before = start.gui_status()
    token = parse_qs(urlsplit(url).fragment)['token'][0]
    if args.execute and (before.get('active') or before['planner'].get('enabled')):
        raise ValueError('已有控制会话，请在原程序或工作台正常结束该会话')
    if args.edge12_forward or args.grasp_only:
        from j6_only import require_capability
        require_capability(start.existing_planner())
    model, urdf_sha = verified_model()
    if preview is not None:
        def preview_right_z():
            joints, _, _ = state_arrays(snapshot(token))
            xyz, _ = model.tcp(joints)
            return xyz[0,1,2]*1000
        preview.start_live(preview_right_z)
    if args.execute:
        from right_approach_motion import execute
        joints, _, grip = state_arrays(before)
        reset_start = dict(epoch=before['epoch'], start_joints_rad=joints.tolist(),
                           gripper_feedback=grip.tolist())
        report['first_reset'] = dict(status='requested')
        began = time.monotonic()
        reset_result = execute(reset_start, token, state_arrays, first_reset=True)
        report['first_reset'].update(reset_result, status='confirmed', elapsed_s=time.monotonic()-began)
        before = snapshot(token)
        reset_joints, _, _ = state_arrays(before)
        reset_xyz, reset_quat = model.tcp(reset_joints)
        report['first_reset'].update(actual_tcp_mm=(reset_xyz[0,1]*1000).tolist(),
                                     actual_wxyz=reset_quat[0,1].tolist(),
                                     actual_joints_rad=reset_joints.tolist())
        print('右臂 first_reset 已确认并停稳，右夹爪打开指令已回显。', flush=True)
    report['perception_batch_id'] = out.name
    report['perception_source'] = 'left_wrist_rgbd_only'
    report['head_camera'] = dict(role='optional_live_preview', gpt_requests=0,
                                 required_for_task=False)
    report['left_wrist_perception'] = {}
    print('100获取155左腕D405 RGB-D，调用GPT识别朝机器人书脊、远端1—2边及书本数量；'
          'Y/X/Z/本数全部来自左腕。头部仅异步实时预览，不参与任务判定。', flush=True)
    from left_wrist_spine import run as perceive_left
    began = time.monotonic()
    try:
        spine = perceive_left(start, credential, out, report['perception_batch_id'],
                              report['left_wrist_perception'], require_far_edge=True)
    except Exception:
        if preview is not None:
            preview.show_left_failure(report['left_wrist_perception'])
        raise
    finally:
        credential = None
        report['task_perception_s'] = time.monotonic()-began
    report['first_left_spine'] = spine
    if spine['epoch'] != before['epoch']:
        raise ValueError('左腕感知期间控制会话改变；未发送Approach')
    report['selection'] = dict(source='left_wrist_gpt', frame_token=spine['frame_token'],
                               robot_spine_reference='robot_facing_spine',
                               robot_spine_polygon_px=spine['polygon_px'])
    if preview is not None:
        preview.show_left_spine(spine)
    print(f'左腕首次书脊基座Z={spine["measured_spine_z_mm"]:.3f}mm；'
          f'Z补偿{spine["base_z_compensation_mm"]:.1f}mm；下降目标Z={spine["target_z_mm"]:.3f}mm。', flush=True)
    measure_approach(before, snapshot(token), model, urdf_sha, report)
    if preview is not None:
        preview.status(f'Y测量来源：{report["y_measurement_source"]}；'
                       f'书脊基座Y={report["measured"]["center_mm"][1]:.3f}mm；'
                       f'offset={report["offset_y_mm"]:+.3f}mm。左腕1—2远边独立用于X。')
    if args.edge12_forward:
        from left_wrist_spine import front_forward
        # Keep the legacy report/CLI name; source explicitly identifies the left RGB-D.
        report['edge12_forward'] = front_forward(report['first_left_spine'], report['perception_batch_id'])
        print(f'左腕前沿基座X中位数={report["edge12_forward"]["base_x_median_mm"]:.3f}mm；'
              f'额外前伸20mm，下拨准备终点X={report["edge12_forward"]["target_tcp_x_mm"]:.3f}mm。', flush=True)
    evidence = out/'measurement.json'
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    y_start = report['right_tcp_mm'][1]
    y_goal = report['measured']['center_mm'][1]
    print(f'右臂起点 Y={y_start:.3f}mm，书中心 Y={y_goal:.3f}mm，offset={report["offset_y_mm"]:+.3f}mm。'
          '保持起点X/Z和末端姿态。', flush=True)
    # Choose the push contact point before moving, without changing the grasp center.
    from preparation_target import select_preparation, apply_preparation
    from pre_sweep_lift import from_left_wrist, STAGE, evidence_for
    lift_config=from_left_wrist(report['first_left_spine'],report['perception_batch_id'],report['epoch'])
    report['pre_sweep_lift_config']=lift_config
    lift_mm=lift_config['lift_mm'];count=lift_config['stack_book_count']
    print(f'首次左腕确认书堆{count}本；下拨前上抬({count}+1)cm={lift_mm:g}mm。',flush=True)
    if preview is not None:preview.status(f'左腕识别{count}本；下拨前上抬({count}+1)cm={lift_mm:g}mm')
    target_position=np.asarray(report['right_tcp_mm'],float)*.001
    target_position[0]=report['edge12_forward']['target_tcp_x_mm']*.001
    target_position[1]=y_goal*.001
    target_position[2]+=lift_mm*.001
    from lift_preflight import request_preflight
    def predict_path(point,seconds):
        scratch=dict(start_joints_rad=report['start_joints_rad'],gripper_feedback=report['gripper_feedback'])
        return request_preflight(model,scratch,point,lift_mm,seconds=seconds,reject=False)
    print('Approach前按实际顺序预检：Y对齐→X前伸→抬升；再从预测抬升终点检查J6准备连接。仅GPU预检，不发布运动。',flush=True)
    try:
        goal,choice=select_preparation(model,report['start_joints_rad'],report['gripper_feedback'],
                                      target_position,report['first_left_spine'],
                                      report['perception_batch_id'],report['epoch'],predict_path=predict_path)
    except ValueError as error:
        if hasattr(error,'preparation_diagnostic'):
            report['preparation_failure']=error.preparation_diagnostic
        raise
    apply_preparation(report,goal,choice)
    geometry=choice['geometry']
    y_goal=report['preparation_y_mm'];target_position[1]=y_goal*.001
    report['axis_endpoint_precheck']=dict(target_joints_rad=goal.tolist(),geometry=geometry,
                                        evidence='Preparation connected from predicted lift endpoint; execution revalidates fresh state')
    report['approach_lift_preflight']=choice['approach_lift_preflight']
    message=(f'拨书准备Y={y_goal:.3f}mm，距书脊中心{choice["center_offset_y_mm"]:+.3f}mm；'
             f'后续抓取仍回到书脊中心Y={report["grasp_center_y_mm"]:.3f}mm。X深度、抬升{lift_mm:g}mm和31°下拨不变。')
    print(message,flush=True)
    if preview is not None:preview.status(message)
    evidence.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Y/X/抬升及准备连接预检通过；执行时仍从各阶段新鲜实测起点重新规划和复核。',flush=True)
    save_report(out,report)
    began = time.monotonic()
    if args.execute:
        report['arrival'] = execute(report, token, state_arrays)
        report.update(status='arrived', motion_s=time.monotonic()-began)
        print('右臂已确认到达本书的拨书准备Y位置。', flush=True)
        if args.edge12_forward:
            current = snapshot(token)
            joints, _, grip = state_arrays(current)
            xyz, quat = model.tcp(joints)
            initial = xyz[0,1]*1000
            target = initial.copy()
            target[0] = report['edge12_forward']['target_tcp_x_mm']
            report['edge12_forward'].update(start_tcp_x_mm=float(initial[0]),
                                             forward_distance_mm=float(target[0]-initial[0]))
            if preview is not None:
                preview.show_front_forward(report['edge12_forward'],float(initial[0]))
            print(f'左腕前沿计算前伸：目标X={target[0]:.3f}mm − 当前X={initial[0]:.3f}mm'
                  f' = {target[0]-initial[0]:.3f}mm；保持Y/Z和姿态。',flush=True)
            if target[0] <= initial[0]:
                raise ValueError('左腕实测远边X终点不在当前+X方向，未发送前伸')
            forward = dict(epoch=current['epoch'], start_joints_rad=joints.tolist(),
                           gripper_feedback=grip.tolist(), right_tcp_mm=initial.tolist(),
                           right_tcp_wxyz=quat[0,1].tolist(),
                           stages=[dict(stage='advance_x',translation_axis='X',
                                        tcp_m_wxyz=np.r_[target*.001,quat[0,1]].tolist())])
            report['forward_evidence'] = forward
            report['forward_arrival'] = execute(forward, token, state_arrays)
            report['status'] = 'edge12_forward_arrived'
            current=snapshot(token)
            joints,_,grip=state_arrays(current)
            xyz,quat=model.tcp(joints)
            lift=evidence_for(current,joints,grip,xyz[0,1],quat[0,1],lift_config=lift_config)
            report[STAGE]=dict(evidence=lift,status='requested')
            save_report(out,report)
            print(f'实测远边X已到位；沿基座+Z抬升{lift_mm:g}mm，保持X/Y和姿态。',flush=True)
            report[STAGE]['arrival']=execute(lift,token,state_arrays)
            report[STAGE]['status']='confirmed'
            print('抬升已确认；重新求解J6轴准备与J7双指对齐，再下拨。',flush=True)
            from j6_geometry import align_goal, downward_goal, j7_align_goal, SWEEP_DEG
            for stage_name in ('j6_axis_align','rotate_j7_fingers_y','rotate_j6_only'):
                current=snapshot(token)
                joints,_,grip=state_arrays(current)
                xyz,quat=model.tcp(joints)
                if stage_name=='j6_axis_align':
                    goal,detail=align_goal(model,joints,grip,
                                          preferred_joints=report['preparation_target']['target_joints_rad'])
                elif stage_name=='rotate_j7_fingers_y':
                    goal,detail=j7_align_goal(model,joints,grip)
                else:
                    goal,detail=downward_goal(model,joints,grip)
                stage=dict(stage=stage_name,target_joints_rad=goal.tolist(),
                           tcp_m_wxyz=np.r_[xyz[0,1],quat[0,1]].tolist())
                if stage_name=='rotate_j6_only':stage['j6_delta_deg']=detail['joint_delta_deg']
                if stage_name=='rotate_j7_fingers_y':stage['j7_delta_deg']=detail['joint_delta_deg']
                motion=dict(epoch=current['epoch'],start_joints_rad=joints.tolist(),
                            gripper_feedback=grip.tolist(),right_tcp_mm=(xyz[0,1]*1000).tolist(),
                            right_tcp_wxyz=quat[0,1].tolist(),stages=[stage])
                report[stage_name]=dict(evidence=motion,geometry=detail,status='requested')
                save_report(out,report)
                print('J6准备/下拨：'+json.dumps(detail,ensure_ascii=False),flush=True)
                report[stage_name]['arrival']=execute(motion,token,state_arrays)
                report[stage_name]['status']='confirmed'
            report['status']='j6_downward_push_completed'
            print(f'仅J6绕基座Y向下{SWEEP_DEG:g}°完成；接续回收60mm、额外抬升50mm及原J7后续抓取。',flush=True)
            from post_sweep_sequence import run as continue_after_sweep
            continue_after_sweep(sys.modules[__name__],model,token,report,preview,out)
        return
    endpoint=np.asarray(report['right_tcp_mm'],float).copy()
    endpoint[1]=y_goal;endpoint[0]=report['edge12_forward']['target_tcp_x_mm']
    endpoint[2]+=lift_mm
    report.update(status='perception_preview_only',requested_endpoint_mm=endpoint.tolist(),
                  motion_validation='Y/X/lift and preparation connection predicted; actual execution requires fresh-state replanning')
    print('完整Y/X/抬升及准备连接预检通过，未发布运动。加 --execute 才执行，执行前重新复核。',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--plan-out', type=Path)
    args = parser.parse_args()
    args.edge12_forward=True
    args.grasp_only=False
    args.push_only=False
    if args.execute and args.plan_out is not None:
        parser.error('--plan-out只用于不运动的预览模式')
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    out = choose_output(start.RUNTIME/'right_approach'/f'{stamp}_{uuid.uuid4().hex[:8]}', args.out)
    plan_out = args.plan_out or out/'curobo.json'
    if plan_out.exists():
        raise ValueError('CuRobo报告已存在，请使用新的输出路径')
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='started', runtime_host=start.PC_IP, execute=args.execute, push_only=args.push_only, grasp_only=args.grasp_only, depth={},
                  selection={'source': 'left_wrist_gpt'})
    preview = None
    try:
        from gpt_segmentation_preview import Preview
        try:
            preview = OptionalPreview(Preview(out), report)
        except Exception as error:
            report['preview_errors'] = [dict(method='init',type=type(error).__name__,reason=str(error))]
            print('预览无法打开，任务诊断仍保留：'+str(error), file=sys.stderr)
        if args.execute:
            from round_control import motion_round
            with motion_round(report, out):
                run(args, out, plan_out, report, preview)
        else:
            run(args, out, plan_out, report, preview)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError,
            http.client.HTTPException, cv2.error) as error:
        report.update(status='rejected', reason=str(error))
        if hasattr(error, 'failure'):
            report['motion_failure'] = error.failure
        raise
    finally:
        if preview is not None:
            preview.close()
        report['total_s'] = time.monotonic() - began
        report_path = save_report(out, report)
        print(f'Approach报告 {report_path}，总耗时 {report["total_s"]:.2f}s。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print('右臂Approach未完成：' + str(error), file=sys.stderr, flush=True)
        sys.exit(2)
