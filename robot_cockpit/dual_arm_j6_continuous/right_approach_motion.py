"""Run one right-arm lateral approach using the existing baseline control session."""

import asyncio
import fcntl
import json
import math
import os
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, ProxyHandler, build_opener
from approach_handoff import check_start, SettledJoints, check_settled_j7, JOINT_START_LIMIT_RAD
from continuous_policy import run_phase as run_segment, phase_count, MAX_PHASE_MM
from pre_sweep_lift import STAGE, stage_lift_mm

import numpy as np
from scipy.spatial.transform import Rotation

import start
from baseline_approach_client import pose as baseline_pose


def validate_pose_sample(sample):
    # Parse the exact gated sample; don't fetch a different frame after validation.
    return baseline_pose(SimpleNamespace(get_current_state=lambda:sample))


def pose(robot):
    while True:
        try:return validate_pose_sample(robot.get_current_state())
        except ValueError as error:
            if (str(error)!='Baseline preflight requires fresh measured feedback' or
                    getattr(robot,'_continuous_wait_feedback',None) is None):raise
            # A sample that crosses 250ms is re-gated, never accepted as fresh.
# Independent trial segment size; baseline cell motion is unchanged.
APPROACH_STEP_MM = MAX_PHASE_MM  # bounded whole phase, not a stop spacing
from robot_cockpit.launcher import book_grasp as app

def preload_motion_dependencies():
    """Import heavy Python/native dependencies before acquiring control; no sockets."""
    from controllor.hkstp_arm import RobotArm
    from cuarm_state import GripperMode, TargetCommand, TargetCommandGripperCommandType, CuroboStatus
    from cart_book_grasp_v3_1_ports import dedicated_planner
    return RobotArm


def set_right_gripper_after_arrival(robot, evidence, cancelled, read_grip, target):
    """Only update gripper joints; preserve the confirmed arm command identity."""
    if target not in (0., 1.):
        raise ValueError('夹爪目标只能为打开0或闭合1')
    operation = '打开' if target == 0. else '闭合'
    if read_grip is None:
        raise RuntimeError(f'缺少新鲜夹爪回显读取接口，未{operation}夹爪')
    if cancelled.is_set():
        raise RuntimeError(f'{operation}夹爪前控制会话已停止')
    current = np.asarray(read_grip(), float)  # Includes current ownership/freshness checks.
    if current.shape != (2,) or not np.isfinite(current).all():
        raise RuntimeError(f'{operation}夹爪前反馈无效')
    if abs(current[0]-evidence['gripper_feedback'][0]) > .001:
        raise RuntimeError(f'{operation}夹爪前左夹爪保持值变化')
    robot.target_command.gripper_joint_command = [[float(evidence['gripper_feedback'][0])], [float(target)]]
    robot.udp.send(robot.target_command)
    began = time.monotonic()
    while time.monotonic()-began < 5:
        if cancelled.is_set():
            raise RuntimeError(f'{operation}夹爪期间控制会话停止')
        grip = np.asarray(read_grip(), float)
        if grip.shape != (2,) or not np.isfinite(grip).all():
            raise RuntimeError(f'{operation}夹爪回显无效')
        if abs(grip[0]-evidence['gripper_feedback'][0]) > .001:
            raise RuntimeError('左夹爪保持值变化')
        if abs(grip[1]-target) <= .001 and time.monotonic()-began >= 2:
            return dict(target=float(target), feedback=float(grip[1]), source='command_echo',
                        elapsed_s=time.monotonic()-began, physical_grasp_confirmed=False)
        time.sleep(.02)
    raise RuntimeError(f'右夹爪{operation}指令未取得回显；停止后续动作')


def close_after_arrival(robot, evidence, cancelled, read_grip):
    return set_right_gripper_after_arrival(robot, evidence, cancelled, read_grip, 1.)


def open_after_reset(robot, evidence, cancelled, read_grip):
    return set_right_gripper_after_arrival(robot, evidence, cancelled, read_grip, 0.)


def open_before_reset(robot, evidence, cancelled, read_grip, joints):
    """Release while commanding the measured hold, never the reset/zero target."""
    from cuarm_state import TargetCommandArmCommandType
    joints=np.asarray(joints,float)
    if joints.shape!=(14,) or not np.isfinite(joints).all():
        raise ValueError('松爪前缺少有效双臂实测关节；不复位')
    robot.target_command.arm_command_type=TargetCommandArmCommandType.JOINT
    robot.target_command.arm_joint_size=[7,7]
    robot.target_command.arm_joint_command=joints.reshape(2,7).copy()
    robot.target_command.description=f'right_trial_release_before_reset:joint_hold:{time.time_ns()}'
    return set_right_gripper_after_arrival(robot,evidence,cancelled,read_grip,0.)


def drive(evidence, cancelled, first_reset, read_grip=None, confirm_settled=None, confirm_start=None, *, client_pool=None):
    if confirm_settled is None or confirm_start is None:
        raise RuntimeError('缺少本段停稳确认接口；未发送运动')
    from cart_book_grasp_v3_1_ports import dedicated_planner
    from controllor.hkstp_arm import RobotArm
    from cuarm_state import GripperMode, TargetCommand, TargetCommandGripperCommandType, CuroboStatus

    class ApproachArm(RobotArm):
        def _check_lower_orientation(self, state, reference, tolerance_rad):
            from motion_failure import check_orientation
            stage = self._trial_evidence.get('stages', [{}])[0].get('stage', 'unknown')
            return check_orientation(self, state, reference, tolerance_rad, stage)

        def _load_configuration(self):
            # The dual-gripper assembly has two inline single-axis grippers.
            # Send their current normalized values as JOINT commands.
            self.target_command = TargetCommand(
                arm_size=2, arm_joint_size=[7, 7], arm_joint_command=[[0.] * 7 for _ in range(2)],
                arm_tool_command=np.asarray([[0., 0., 0., 1., 0., 0., 0.]] * 2),
                gripper_command_type=TargetCommandGripperCommandType.JOINT,
                gripper_size=2, gripper_joint_size=[1, 1],
                gripper_mode_command=[GripperMode.OPEN] * 2,
                gripper_joint_command=[[value] for value in evidence['gripper_feedback']],
            )

        def get_current_state(self):
            if cancelled.is_set():
                raise RuntimeError('Approach控制会话已停止')
            wait=getattr(self,'_continuous_wait_feedback',None)
            sample=wait() if wait is not None else super().get_current_state()
            tracking=getattr(self,'_motion_tracking',None)
            if tracking is not None:tracking.observe_monitor(sample,getattr(self,'_tracking_context',{}))
            return sample

        def _continuous_raw_state(self):
            return super().get_current_state()

        def _wait_for_robot_stop(self, *args, **kwargs):
            if self._trial_evidence.get('stages', [{}])[0].get('stage') in ('wrist_z_servo', 'lower_z5'):
                kwargs['position_tolerance_m'] = .0005
                kwargs['require_fresh_stable'] = True
            return super()._wait_for_robot_stop(*args, **kwargs)

    ports = SimpleNamespace(TARGET_LOCAL_PORT=10175, TARGET_REMOTE_PORT=start.TARGET_PORT)
    with dedicated_planner(ports):
        def create_robot():
            return ApproachArm(target_host='127.0.0.1', target_local_port=ports.TARGET_LOCAL_PORT,
                               target_remote_port=ports.TARGET_REMOTE_PORT)
        robot = client_pool.acquire(create_robot, evidence) if client_pool is not None else create_robot()
        robot._trial_evidence = evidence
        try:
            if robot.udp is None:
                raise RuntimeError('无法连接CuRobo任务端口，请结束其他抓取程序')
            deadline = time.monotonic() + 4
            while True:
                try:
                    _, joints, xyz, _ = pose(robot)
                    break
                except (ValueError, TypeError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('CuRobo缺少新鲜实测TCP反馈') from None
                    time.sleep(.05)
            # Setup/ownership waits are complete. Hold and collect a new stable
            # RT window before binding translation intents to the actual start.
            stable = confirm_start()
            paired_deadline = time.monotonic()+1.
            while True:
                _, joints, xyz, start_angles = pose(robot)
                error = float(np.max(np.abs(joints-np.asarray(stable['joints_rad']))))
                if error <= .001:
                    break
                if time.monotonic() >= paired_deadline:
                    evidence['start_channel_failure'] = dict(joint_delta_rad=error, limit_rad=.001)
                    raise RuntimeError('停稳RT与Planner起点反馈不一致；未发送运动')
                time.sleep(.025)
            if read_grip is None or not np.array_equal(np.asarray(read_grip()), evidence['gripper_feedback']):
                raise RuntimeError('停稳后夹爪回显变化；未发送运动')
            if np.max(np.abs(joints - evidence['start_joints_rad'])) > JOINT_START_LIMIT_RAD:
                raise RuntimeError('当前起点与记录的关节起点偏移超过0.5度；未发送运动')
            if not first_reset:
                from motion_start import bind_settled_start
                quat = Rotation.from_euler('xyz', start_angles, degrees=True).as_quat()[[3,0,1,2]]
                bind_settled_start(evidence, joints, xyz, quat)
            if first_reset:
                from trial_reset_pose import load_target
                reset_angles,reset_config=load_target()
                robot.right_safe_joint_deg=reset_angles
                evidence['reset_pose_config']=reset_config
                if 'target_right_joint_deg' in evidence:
                    reset_target = np.asarray(evidence['target_right_joint_deg'],float)
                    if reset_target.shape != (7,) or not np.isfinite(reset_target).all():
                        raise ValueError('本轮记录的first_reset关节目标无效')
                    robot.right_safe_joint_deg = reset_target.copy()
                print('独立first_reset：使用固定-X60mm、-Y150mm准备位；不累加偏移。',flush=True)
                print('first_reset：先打开右夹爪，双臂保持当前关节；等待回显和2秒稳定。', flush=True)
                opened = open_before_reset(robot,evidence,cancelled,read_grip,joints)
                evidence['release_before_reset']=opened
                confirm_start()
                if cancelled.is_set():
                    raise RuntimeError('松爪后控制已停止；不执行复位')
                print('右夹爪打开指令已回显；开始右臂 first_reset，保持左臂和左夹爪。',flush=True)
                if robot.control_joint_position(left_pos=False, right_pos=True, action='first_reset', force=True) is not True:
                    raise RuntimeError('右臂 first_reset 未确认完成，停止本次任务')
                return confirm_settled(dict(target_right_joint_deg=robot.right_safe_joint_deg.tolist(), right_gripper_open=opened))
            initial = np.asarray(evidence['right_tcp_mm'])
            target = np.asarray(evidence['stages'][0]['tcp_m_wxyz'][:3]) * 1000
            reference = Rotation.from_quat(np.asarray(evidence['right_tcp_wxyz'])[[1, 2, 3, 0]])
            if evidence['stages'][0]['stage'] not in ('rotate_j6_only','rotate_j7_fingers_y','rotate_j7_only','j6_axis_align') and np.linalg.norm(xyz - initial) > 1.:
                raise RuntimeError('当前起点与识别时的实测起点不一致，请重跑Approach')
            angles = reference.as_euler('xyz', degrees=True)
            stage = evidence['stages'][0]
            hold = stage.get('hold_pose_mm_wxyz')
            if hold is not None:
                hold = np.asarray(hold, float)
                if stage['stage'] != 'wrist_z_servo' or hold.shape != (7,) or not np.isfinite(hold).all():
                    raise ValueError('Invalid wrist Z hold reference')
                hold_rotation = Rotation.from_quat(hold[3:][[1,2,3,0]])
                if (np.linalg.norm(initial[:2]-hold[:2])>3. or
                        (reference.inv()*hold_rotation).magnitude()>np.deg2rad(1.)):
                    raise ValueError('视觉伺服固定轴或姿态偏离参考，暂停')
                reference = hold_rotation
                angles = reference.as_euler('xyz', degrees=True)
            if stage['stage'] == 'rotate_j7_only':
                from j7_only import ACTION
                goal = np.asarray(stage['target_joints_rad'], float)
                if (goal.shape != (14,) or not np.isfinite(goal).all()
                        or not np.allclose(goal[:13], evidence['start_joints_rad'][:13], atol=1e-8, rtol=0)
                        or not np.isclose(abs(goal[13]-evidence['start_joints_rad'][13]),np.pi/2,atol=1e-8)):
                    raise ValueError('仅J7指令不得改变其他关节，且角度必须为90度')
                robot.right_safe_joint_deg = np.rad2deg(goal[7:])
                if robot.control_joint_position(left_pos=False,right_pos=True,action=ACTION,force=True) is not True:
                    raise RuntimeError('仅J7旋转未确认；不继续后退')
                _, actual_joints, actual, actual_angles = pose(robot)
                # Completion is not proof of settled physical feedback. Record
                # this sample, then apply the SAME tolerances after settling.
                evidence['j7_completion_sample'] = dict(
                    actual_joints_rad=actual_joints.tolist(), target_joints_rad=goal.tolist(),
                    delta_rad=(actual_joints-goal).tolist(), decision='await_continuous_settling')
                return confirm_settled(dict(actual_tcp_mm=actual.tolist(),actual_angles_deg=actual_angles.tolist(),
                            actual_joints_rad=actual_joints.tolist(),j7_delta_deg=stage['j7_delta_deg'],
                            rotation_mode='right_J7_joint_only'))
            if stage['stage'] == 'j6_axis_align':
                from j6_geometry import axis_error, j7_align_goal
                from backend.board_servo.model import Model
                model=Model(start.ROOT/'backend/board_servo')
                goal=np.asarray(stage['target_joints_rad'],float)
                if (goal.shape!=(14,) or not np.isfinite(goal).all()
                        or not np.allclose(goal[:7],joints[:7],atol=.002,rtol=0)):
                    raise ValueError('J6准备姿态不得改变左臂')
                j7_align_goal(model,goal,evidence['gripper_feedback'])
                robot.right_safe_joint_deg=np.rad2deg(goal[7:])
                if robot.control_joint_position(left_pos=False,right_pos=True,action='right_trial_j6_axis_align',force=True) is not True:
                    raise RuntimeError('J6准备姿态规划/到位未确认；不执行下拨')
                result=confirm_settled(dict(target_joints_rad=goal.tolist(),rotation_mode='prepare_J6_base_Y'))
                _,actual,_,_=pose(robot)
                j7_align_goal(model,actual,evidence['gripper_feedback'])
                return result
            if stage['stage']=='rotate_j7_fingers_y':
                from j6_only import J7_ACTION
                from j6_geometry import j7_align_goal, check_finger_alignment
                from backend.board_servo.model import Model
                model=Model(start.ROOT/'backend/board_servo')
                goal,detail=j7_align_goal(model,joints,evidence['gripper_feedback'])
                stage['target_joints_rad']=goal.tolist()
                evidence['j7_finger_alignment']=detail
                print('仅J7调整双指方向：'+json.dumps(detail,ensure_ascii=False),flush=True)
                if abs(detail['joint_delta_deg'])>0.01:
                    robot.right_safe_joint_deg=np.rad2deg(goal[7:])
                    if robot.control_joint_position(left_pos=False,right_pos=True,action=J7_ACTION,force=True) is not True:
                        raise RuntimeError('仅J7双指对齐未确认；不执行J6下拨')
                result=confirm_settled(dict(target_joints_rad=goal.tolist(),rotation_mode='right_J7_fingers_parallel_Y'))
                _,actual,_,_=pose(robot)
                check_finger_alignment(model,actual)
                return result
            if stage['stage'] == 'rotate_j6_only':
                from j6_only import ACTION
                from j6_geometry import SWEEP_DEG
                goal = np.asarray(stage['target_joints_rad'], float)
                if (goal.shape != (14,) or not np.isfinite(goal).all()
                        or not np.allclose(goal[np.arange(14)!=12], np.asarray(evidence['start_joints_rad'])[np.arange(14)!=12], atol=1e-8, rtol=0)
                        or not np.isclose(abs(goal[12]-evidence['start_joints_rad'][12]),np.deg2rad(SWEEP_DEG),atol=1e-8)):
                    raise ValueError(f'仅J6指令不得改变其他关节，且角度必须为{SWEEP_DEG:g}度')
                from backend.board_servo.model import Model
                from j6_geometry import downward_goal
                model=Model(start.ROOT/'backend/board_servo')
                exact,detail=downward_goal(model,joints,evidence['gripper_feedback'])
                # Rebind only J6 delta to the fresh settled sample; hold all others exactly.
                goal=exact
                stage['target_joints_rad']=goal.tolist()
                stage['j6_delta_deg']=detail['joint_delta_deg']
                evidence['j6_measured_sweep']=detail
                robot.right_safe_joint_deg = np.rad2deg(goal[7:])
                if robot.control_joint_position(left_pos=False,right_pos=True,action=ACTION,force=True) is not True:
                    raise RuntimeError('仅J6旋转未确认；不继续后退')
                _, actual_joints, actual, actual_angles = pose(robot)
                # Completion is not proof of settled physical feedback. Record
                # this sample, then apply the SAME tolerances after settling.
                evidence['j6_completion_sample'] = dict(
                    actual_joints_rad=actual_joints.tolist(), target_joints_rad=goal.tolist(),
                    delta_rad=(actual_joints-goal).tolist(), decision='await_continuous_settling')
                return confirm_settled(dict(actual_tcp_mm=actual.tolist(),actual_angles_deg=actual_angles.tolist(),
                            actual_joints_rad=actual_joints.tolist(),j6_delta_deg=stage['j6_delta_deg'],
                            rotation_mode='right_J6_joint_only'))
            axis = evidence['stages'][0].get('translation_axis', 'Y')
            if axis not in ('X', 'Y', 'Z'):
                raise ValueError('Unsupported approach translation axis')
            component = {'X':0, 'Y':1, 'Z':2}[axis]
            fixed = [i for i in range(3) if i != component]
            fixed_reference = hold[:3] if hold is not None else initial
            if not np.allclose(target[fixed], fixed_reference[fixed], atol=1e-6):
                raise ValueError('单轴平移目标改变了固定坐标')
            offset = target[component] - initial[component]
            withdrawing = stage['stage'] == 'withdraw_x'
            returning = stage['stage'] == 'withdraw_to_reset_x'
            if withdrawing and (axis != 'X' or not np.isclose(offset,-60.,atol=1e-6)):
                raise ValueError('后退仅允许沿-X固定60mm')
            if returning and (axis != 'X' or offset > 3. or
                              not np.isclose(target[0], stage['first_reset_x_mm'],atol=1e-6,rtol=0)):
                raise ValueError('退回准备位X必须沿-X且终点等于本轮保存的first_reset X')
            if stage.get('downward_only') and (axis!='Z' or offset>0):
                raise ValueError('本轮腕部Z伺服仅允许下降')
            if axis == 'Z':
                expected_offset = {'lower_z5': -5., 'raise_z50': 50.,
                                   'post_grasp_raise_z200': 200., 'post_grasp_lower_z200': -200.}.get(stage['stage'])
                if stage['stage']==STAGE:expected_offset=stage_lift_mm(stage)
                if stage['stage'] == 'cached_top_lower_z':
                    top = float(stage['target_top_z_mm'])
                    if not np.isfinite(top) or offset > 0 or not np.isclose(target[2], top, atol=1e-6, rtol=0):
                        raise ValueError('缓存顶面下降只允许−Z且终点必须等于首次实测Z')
                elif stage['stage'] == 'wrist_z_servo':
                    if not np.isfinite(offset) or abs(offset)>3.5:
                        raise ValueError('腕部Z伺服单步不得超过3.5mm（含探测回程误差）')
                elif expected_offset is None or not np.isclose(offset,expected_offset,atol=1e-6,rtol=0):
                    raise ValueError('Z位移不符合本动作定义（下压5mm、回拨抬升50mm、抓后抬升/下降200mm）')
            if axis == 'X' and not (withdrawing or returning) and offset <= 0:
                raise ValueError('前伸目标必须位于当前+X方向')
            minimum = 1e-6 if stage['stage'] in ('wrist_z_servo', 'lower_z5') else 3.
            print('连续轨迹：整段GPU规划、一次发布；中间控制点不停，末端确认到位。', flush=True)
            count = phase_count(offset, minimum)
            attempts = evidence.setdefault('segment_attempts', [])
            completed_segments = 0
            attempt_start = {}
            def attempt_point(point):
                from planning_receipt import new_request, tagged, verify_rejection
                # A fresh per-attempt failure marker avoids retrying an old failure.
                robot.last_motion_failure = None
                _, q_before, _, _ = pose(robot)
                grip_before = np.asarray(read_grip(), float)
                attempt_start.update(joints=np.array(q_before, copy=True), grip=grip_before.copy())
                request = new_request(start.existing_planner())
                attempt_start['request'] = request
                robot._tracking_context=dict(stage=stage['stage'],request_id=request['request_id'],
                    action={'X':'book_plug_x','Y':'book_plug_y','Z':'book_plug_z'}[axis],
                    target_tcp_m=[float(v)*.001 for v in point],fixed_axes=[i for i in range(3) if i!=component])
                if getattr(robot,'_motion_tracking',None) is not None:
                    robot._motion_tracking.emit(dict(event='motion_request',context=robot._tracking_context.copy()))
                print(f'Approach目标 {axis}={point[component]:.3f}mm。', flush=True)
                send = robot.udp.send
                robot.udp.send = lambda command: send(tagged(command, request))
                try:
                    ok = robot.master_control(right_arm=[*point, *angles],
                        action={'X':'book_plug_x','Y':'book_plug_y','Z':'book_plug_z'}[axis], object_category='book')
                finally:
                    robot.udp.send = send
                failure = getattr(robot, 'last_motion_failure', None)
                if isinstance(failure, dict):
                    failure = dict(failure)
                    if getattr(robot,'_motion_tracking',None) is not None:
                        failure['tracking']=robot._motion_tracking.status()
                    proof = verify_rejection(request, failure, robot._status_value(CuroboStatus.PLANNING_FAILED),
                                             robot._status_value(CuroboStatus.IK_FAILED), legacy_status_flag=True)
                    failure['legacy_execution_started'] = failure.get('execution_started')
                    failure['request_id'] = request['request_id']
                    # For this entry, require authoritative evidence even when
                    # the legacy client happened to report execution_started=False.
                    failure['execution_started'] = (False if proof is not None
                        else True if failure.get('execution_started') is True else None)
                    failure['planning_receipt'] = proof
                    if proof is not None:
                        failure['planner_diagnostics'] = proof.get('diagnostics')
                        local = (proof.get('diagnostics') or {}).get('lift_continuity') or {}
                        if local.get('rejection_reasons'):
                            failure['reason'] = 'lift_continuity_rejected'
                            failure['rejection_reasons'] = local['rejection_reasons']
                    if proof is not None and proof.get('retry_reason') == 'local_lift_ik_failed_before_publication':
                        failure['reason'] = 'local_lift_ik_failed'
                        failure['solver_status'] = 'MotionGenStatus.IK_FAIL'
                    attempt_start['failure'] = failure
                    if proof is not None:
                        print('当前请求已由Planner确认：规划拒绝、轨迹未发布；本连续版本保持，不拆段执行。', flush=True)
                return ok is True, dict(failure) if isinstance(failure, dict) else None

            def confirm_retry_hold():
                from planning_receipt import verify_rejection
                result = confirm_settled(dict(retry_hold=True))
                if verify_rejection(attempt_start['request'], attempt_start['failure'],
                                    robot._status_value(CuroboStatus.PLANNING_FAILED),
                                    robot._status_value(CuroboStatus.IK_FAILED)) is None:
                    raise RuntimeError('停稳期间规划未执行证据失效；不细分重试')
                _, q_after, p_after, angles_after = pose(robot)
                grip_after = np.asarray(read_grip(), float)
                delta = np.asarray(q_after)-attempt_start['joints']
                orientation_error = (reference.inv()*Rotation.from_euler('xyz', angles_after, degrees=True)).magnitude()
                detail = dict(joint_delta_rad=delta.tolist(), actual_tcp_mm=np.asarray(p_after).tolist(),
                              before_gripper=attempt_start['grip'].tolist(), after_gripper=grip_after.tolist(),
                              orientation_error_deg=float(np.rad2deg(orientation_error)))
                evidence['retry_hold_latest'] = detail
                if (np.max(np.abs(delta)) > .002
                        or not np.array_equal(grip_after, attempt_start['grip'])
                        or np.linalg.norm(np.asarray(p_after)[fixed]-target[fixed]) > 3.
                        or orientation_error > np.deg2rad(1.)):
                    raise RuntimeError('规划拒绝后实测保持状态变化；不细分重试')
                return dict(feedback=result['settled_feedback'], **detail)

            segment_start = initial.copy()
            segment_start[fixed] = target[fixed]
            for index in range(1, count + 1):
                point = initial.copy()
                point[fixed] = target[fixed]
                point[component] = initial[component] + offset * index / count
                print(f'连续Approach，右臂终点 {axis}={point[component]:.3f}mm。', flush=True)
                completed_segments += run_segment(segment_start, point, component, attempt_point,
                    confirm_retry_hold, attempts, robot._status_value(CuroboStatus.PLANNING_FAILED))
                segment_start = point.copy()
            _, actual_joints, actual, actual_angles = pose(robot)
            angle = (reference.inv() * Rotation.from_euler('xyz', actual_angles, degrees=True)).magnitude()
            arrival_mm = .5 if stage['stage'] in ('wrist_z_servo', 'lower_z5') else 3.
            if (np.linalg.norm(actual - target) > arrival_mm or angle > np.deg2rad(1.)
                    or np.max(np.abs(actual_joints[:7] - joints[:7])) > .002):
                raise RuntimeError('实测TCP或左臂保持误差超限')
            result = dict(actual_tcp_mm=actual.tolist(), segments=completed_segments, original_segments=count,
                          max_phase_mm=MAX_PHASE_MM, execution_policy="whole_phase", segment_attempts=attempts)
            if stage.get('close_right_gripper_after_arrival'):
                if stage['stage'] != 'advance_x' or axis != 'X' or count == 0:
                    raise ValueError('仅确认完成前伸后允许闭爪')
                result['gripper'] = close_after_arrival(robot, evidence, cancelled, read_grip)
            return confirm_settled(result)
        finally:
            if client_pool is None:
                robot.shutdown()


async def control(token, url, evidence, read_arrays, first_reset):
    cancelled = threading.Event()
    worker = None
    async with app.websockets.connect(url, proxy=None, compression=None,
                                     open_timeout=5, ping_interval=2, close_timeout=1) as connection:
        ws = app.JsonSocket(connection)
        await ws.send_json(dict(token=token))
        arm = app.ArmSession(ws)
        try:
            await arm.wait(lambda s: s.get('fresh') and s.get('planner', {}).get('connected'), 5, '双夹爪Planner反馈')

            def guard():
                check_start(evidence, arm.state, read_arrays)

            if arm.state.get('active') or arm.state['planner'].get('enabled'):
                raise RuntimeError('已有活动控制，请先正常退出原控制会话')
            guard()
            await arm.enable(guard)
            def read_grip():
                arm.check()
                arm.check_owned()
                return read_arrays(arm.state)[2]

            async def settle_result(result, *, at_start=False):
                print('连接就绪；确认执行前连续停稳并读取新起点。' if at_start else '本段到位后确认连续停稳，再释放会话并进入下一段。', flush=True)
                window = SettledJoints(app.StationaryWindow(), read_arrays, arm.state['epoch'])
                def settled(state):
                    arm.check_owned()
                    return window.observe(state, time.monotonic())
                try:
                    await arm.wait(settled, app.TIMEOUT_SECONDS, '执行前连续停稳' if at_start else '本段完成后连续停稳')
                    if not at_start:
                        check_settled_j7(evidence, window.latest['joints_rad'])
                    result['settled_feedback'] = window.latest
                finally:
                    evidence['start_settling' if at_start else 'completion_settling'] = window.latest
                return result
            loop = asyncio.get_running_loop()
            def confirm_settled(result, *, at_start=False):
                future = asyncio.run_coroutine_threadsafe(settle_result(result, at_start=at_start), loop)
                try:
                    return future.result(timeout=app.TIMEOUT_SECONDS+2.)
                except BaseException:
                    future.cancel()
                    raise

            worker = asyncio.create_task(asyncio.to_thread(
                drive, evidence, cancelled, first_reset, read_grip, confirm_settled,
                lambda: confirm_settled({}, at_start=True)['settled_feedback']))
            while not worker.done():
                arm.check()
                arm.check_owned()
                await asyncio.sleep(.05)
            arm.check_owned()
            result = worker.result()
            return result
        finally:
            cancelled.set()
            try:
                await arm.close()
            finally:
                if worker is not None:
                    await asyncio.gather(worker, return_exceptions=True)


def execute(evidence, token, read_arrays, *, first_reset=False):
    from round_control import current
    session = current()
    if session is not None:
        return session.execute(evidence, token, read_arrays, first_reset)
    pid = start.existing_planner()
    if pid is None:
        raise RuntimeError('先运行 bash /mnt/ssd/user/V/right_arm_j6_continuous_start.sh 准备J6专用CuRobo')
    start.wait_planner(pid, None)
    agent = start.V / 'dual_v2_2_agent'
    sys.path.insert(0, str(agent / 'RobotRAM'))
    os.environ['CUARM_CONFIG_ROOT'] = str(start.ROOT / 'configuration')
    app.RUNTIME.mkdir(parents=True, exist_ok=True)
    with (app.RUNTIME / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有抓取会话，请先退出原程序') from None
        app.refuse_duplicate_agent()
        with app.dedicated_control(remote_port=8810) as url:
            result = asyncio.run(control(token, url, evidence, read_arrays, first_reset))
            # close() sends STOP asynchronously. Observe its completed release
            # before the caller takes the next stage's epoch/position snapshot.
            deadline = time.monotonic()+3.
            opener = build_opener(ProxyHandler({}))
            request = Request(start.GUI+'/api/state', headers={'X-Arm-Token': token})
            while time.monotonic() < deadline:
                with opener.open(request, timeout=1.) as response:
                    state = json.load(response)
                read_arrays(state)  # Fresh RT and model checks remain mandatory.
                if not state.get('active') and not state.get('planner', {}).get('enabled'):
                    result['released_epoch'] = state.get('epoch')
                    return result
                time.sleep(.05)
            raise RuntimeError('本段控制会话释放未确认；不记录下一段起点、不重新接管')
