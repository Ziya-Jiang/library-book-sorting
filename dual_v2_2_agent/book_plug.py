"""Opt-in two-pass insertion, composed with the existing V3.4 grasp runtime."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from book_grasp_post_lift import _fresh_right_pose
from book_plug_slots import CONFIG, load_config, build_payload, select_slot
from cart_book_grasp_v3_4_batch import BatchBookSelectionAgent, BatchCartLoop
from cart_book_grasp_v2_1_selection import load_config as selection_config


def preview_targets(start, rotation, center_mm, parameters, delta_z_mm=0., book_surface_x_mm=None):
    from cart_book_grasp_v1_3_depth import forward_parameters
    from cart_book_grasp_v1_3_visualization import forward_targets
    aligned = np.asarray(start, float).copy(); aligned[1] = center_mm[1]
    compensated = aligned+[0., 0., delta_z_mm]
    parameters = forward_parameters(compensated, rotation, book_surface_x_mm, parameters)
    forward = forward_targets(compensated, parameters)
    grasp = forward[-1] if forward else compensated
    lifted = grasp.copy(); lifted[2] += load_config()['transfer_lift_mm']
    withdrawn = lifted.copy(); withdrawn[0] = start[0]
    points = [np.asarray(start), aligned, compensated, *forward, lifted, withdrawn]
    labels = ['start', 'align Y', 'depth Z compensation', *['grasp approach']*len(forward),
              'lift Z before extraction', 'extract to initial X / head slot observation next']
    return dict(points_mm=np.array(points), labels=labels, grasp_mm=grasp,
                rotation_deg=np.asarray(rotation), parameters=parameters)


def transfer_tasks(tasks):
    if not tasks or not any(t.get('book_grasp_mode') == 'cart+' for t in tasks):
        return tasks
    selected = [dict(t) for t in tasks if t.get('book_grasp_mode') == 'cart+'
                and t.get('arm') == 'right' and t.get('action') in
                ('move_above_grasp', 'move_down_to_grasp', 'close_gripper')]
    if [t['action'] for t in selected] != ['move_above_grasp', 'move_down_to_grasp', 'close_gripper']:
        raise ValueError('插槽需要完整且唯一的右臂Y对齐、视觉前伸、合爪任务')
    selected.append(dict(selected[-1], action='book_plug_transfer', target_position=None,
                         target_rotation=None, description='抬Z→抽回初始X→头部找槽→横移→槽口→插入→降Z→松爪→抽回→复位'))
    print('插槽实际任务：保留Y对准、腕部伺服抓书；合爪后按配置抬Z，保持此高度抽回、横移和插入，最终X处竖直放下再松爪。', flush=True)
    return selected


class PlugSelectionAgent(BatchBookSelectionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = -1
        self.reset_x_mm = None
        self.current_book = None
        self.loop = None
        self.visualization = None

    @property
    def direction_label(self):
        return 'right to left / move -Y' if self.direction == -1 else 'left to right / move +Y'

    def check_ready(self):
        super().check_ready()
        load_config()

    def ordered_numbers(self, items):
        return [item['number'] for item in (reversed(items) if self.direction == -1 else items)]

    def restart_batch(self):
        super().restart_batch()
        self.direction = -1
        self.reset_x_mm = None
        self.current_book = None

    def begin_left_pass(self):
        if self.direction != -1 or not self.finished:
            raise ValueError('右移尚未全部完成，不能开始左移')
        self.last_decision['next_pass'] = 'fresh_head_left_to_right'
        self._write()
        self.clear_batch_snapshot()
        self.direction = 1
        self.reset_x_mm = None
        self.current_book = None

    def _write(self):
        if self.batch_output is None:
            return
        self.last_decision.update(selection_mode='plug', batch_status=self.batch_status,
            pass_direction=self.direction, order=self.batch_order,
            completed_numbers=list(self.completed_numbers), next_index=self.batch_index,
            book_frame_policy='one_snapshot_per_pass', slot_frame_policy='live_after_each_extraction',
            scene_comparison=False)
        payload = json.dumps(self.last_decision, ensure_ascii=False, indent=2)
        for name in ('decision.json', 'batch.json'):
            (self.batch_output/name).write_text(payload)

    def select_instances(self, *args, **kwargs):
        # This runs only after the inherited first_reset-before-perception.
        # It precedes J7 prealignment and visual grasping; do not infer X from
        # a future grasp pose or a constant from a past experiment.
        self.reset_x_mm = float(_fresh_right_pose(self.robot)[0][0])
        item = super().select_instances(*args, **kwargs)
        self.current_book = deepcopy(item)
        self.last_decision['first_reset_measured_x_mm'] = self.reset_x_mm
        self._write()
        return item

    def find_slot(self, robot, config, checkpoint, log, *, height_mm=None):
        from cart_book_grasp_v2_gpt import GPTBookClient, client_options
        from cart_book_grasp_v3_2_head_fallback import load_config as model_config
        if self.current_book is None or self.snapshot is None or not callable(getattr(self, 'live_capture', None)):
            raise ValueError('缺少当前书身份或实时头部相机')
        output = self.batch_output/('slot_'+uuid.uuid4().hex)
        output.mkdir()
        anchor, orientation = _fresh_right_pose(robot)
        selection_pose = anchor.copy()
        if height_mm is not None:
            if not np.isfinite(height_mm) or abs(height_mm-anchor[2]) > 3.:
                raise ValueError('找槽高度与抬升后保持高度不一致')
            selection_pose[2] = height_mm

        def guarded():
            checkpoint()
            xyz, rot = _fresh_right_pose(robot)
            if np.linalg.norm(xyz-anchor) > 3. or np.rad2deg((orientation.inv()*rot).magnitude()) > 1.:
                raise ValueError('找槽期间末端离开保持位置；本次槽口观测作废')

        record = dict(status='pending', pass_direction=self.direction, parameters=config,
                      held_number=self.current_book['number'], start_xyz_mm=anchor.tolist(),
                      fixed_height_mm=float(selection_pose[2]), camera_source='head_rgbd')
        try:
            guarded()
            camera = self.observation_camera
            old_token = camera.frame_token
            rgb, depth = self.live_capture()  # Deliberately NOT camera.capture_image().
            if camera.frame_token == old_token:
                raise ValueError('空槽取帧未更新；不能使用缓存书本图像')
            intrinsic, c2w = camera.K.copy(), camera.c2w.copy()
            rgb, depth = rgb.copy(), depth.copy()
            record.update(frame_token=str(camera.frame_token), captured_unix_s=camera.captured_unix_s,
                          K=intrinsic.tolist(), c2w=c2w.tolist())
            Image.fromarray(rgb).save(output/'slot_rgb.png')
            np.save(output/'slot_depth.npy', depth)
            if self.visualization is not None:
                self.visualization.safe('log_image', 'book/images/plug_slot', rgb)
                self.visualization.safe('flush_book')
            x0, y0, x1, y1 = map(int, self.current_book['box'])
            reference = self.snapshot['rgb'][y0:y1, x0:x1].copy()
            if not reference.size:
                raise ValueError('当前书参考图无效')
            model = model_config()
            scene_id = output.name
            record.update(model=model['model'], scene_id=scene_id)
            log(dict(event='slot_observation', output=str(output), direction=self.direction,
                     policy='new_head_frame_after_extraction', camera_source='head_rgbd', model=model['model']))
            record.update(provider=model.get('provider','openai'), model=model['model'])
            response = GPTBookClient(**client_options(model)).select(
                build_payload(rgb, reference, scene_id, model, self.current_book.get('description', ''),
                              transfer_lift_mm=config['transfer_lift_mm']),
                model['timeout_seconds'], guarded)
            record['response'] = response
            guarded()
            bounds = selection_config(self.config_path, require_semantic_model=False)['head_base_bounds_mm']
            slot = select_slot(response, scene_id, depth, intrinsic, c2w, selection_pose, self.direction,
                               config, bounds, model['min_confidence'])
            record.update(status='selected', selected=slot)
            image = rgb.copy()
            uv = np.rint(np.asarray(slot['mouth_quad'])*[rgb.shape[1]-1, rgb.shape[0]-1]/1000.).astype(np.int32)
            cv2.polylines(image, [uv], True, (20, 240, 50), 3)
            cv2.putText(image, f"SLOT X={slot['mouth_x_mm']:.1f} Y={slot['y_mm']:.1f} | insert {config['insertion_mm']:g}mm",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 240, 50), 2)
            Image.fromarray(image).save(output/'slot_selected.png')
            if self.visualization is not None:
                self.visualization.safe('log_image', 'book/images/plug_slot', image)
                self.visualization.safe('log_json_document', 'book/plug_slot', slot)
                self.visualization.safe('flush_book')
            log(dict(event='slot_selected', **slot))
            return slot
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            (output/'slot.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))


class PlugCartLoop(BatchCartLoop):
    completion_message = '右移、左移两轮插槽全部完成，结束；不自动重复。'
    startup_message = ('插槽两轮：先从右到左抓书并向−Y空槽插入；完成后重新识别，从左到右向+Y空槽插入。'
                       '抓后抬Z、头部找槽、等高插入、最终X处降Z再松爪；失败保持，回车复位重启；s退出，p松爪复位。')

    def __init__(self, selector):
        super().__init__(selector)
        selector.loop = self

    def advance_pass(self):
        if self.selector.direction == -1:
            self.selector.begin_left_pass()
            print('右移全部完成。清除首轮书本缓存，重新识别后从左到右执行左移。', flush=True)
            return True
        return False


def move_axis(robot, axis, goal, orientation, config, checkpoint, log, stage, fixed_position=None):
    """Keep the other two coordinates anchored for the entire stage."""
    start, _ = _fresh_right_pose(robot)
    if axis not in (0, 1, 2) or not np.isfinite(goal):
        raise ValueError('插槽只允许基座X/Y/Z单轴移动')
    distance = float(goal-start[axis])
    anchor = start.copy() if fixed_position is None else np.asarray(fixed_position, float).copy()
    fixed = [i for i in range(3) if i != axis]
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or np.linalg.norm((start-anchor)[fixed]) > 3.:
        raise ValueError('插槽阶段接续固定轴漂移超过3mm：'+stage)
    limit = (config['max_x_travel_mm'], config['max_lateral_mm'], config['transfer_lift_mm']+3.)[axis]
    if abs(distance) > limit:
        raise ValueError('插槽单轴位移超出配置范围：'+stage)
    step = (config['vertical_step_mm'] if axis == 2 else
            config['insertion_step_mm'] if stage in ('mouth', 'insert') else config['transfer_step_mm'])
    count = max(1, math.ceil(abs(distance)/step))
    for index in range(1, count+1):
        checkpoint()
        target = anchor.copy(); target[axis] = start[axis]+distance*index/count
        log(dict(event='command', stage=stage, axis='XYZ'[axis], target_mm=target.tolist(),
                 target_wxyz=orientation.as_quat()[[3, 0, 1, 2]].tolist(), segment=index, segments=count))
        pose = np.r_[target, orientation.as_euler('xyz', degrees=True)].tolist()
        if robot.master_control(right_arm=pose, action='book_plug_'+'xyz'[axis],
                                object_category='book') is not True:
            raise ValueError('插槽运动未确认：'+stage)
        actual, measured = _fresh_right_pose(robot)
        error = float(np.linalg.norm(actual-target))
        angle = float(np.rad2deg((orientation.inv()*measured).magnitude()))
        if error > 3. or angle > 1.:
            raise ValueError(f'插槽到位未通过：{stage}, position={error:.3f}mm, angle={angle:.3f}deg')
        log(dict(event='confirmed', stage=stage, actual_mm=actual.tolist(), error_mm=error, angle_deg=angle))


def execute_transfer(selector, robot, task, task_index, types, execute, **kwargs):
    config = load_config()  # One coherent hot-loaded configuration per held book.
    if selector.reset_x_mm is None or selector.batch_output is None:
        raise ValueError('缺少本书first_reset实测X，不能抽回')
    path = selector.batch_output/'transfer.jsonl'

    def log(event):
        event = dict(unix_s=time.time(), pass_direction=selector.direction,
                     book_number=selector.current_book['number'], **event)
        print('插槽：'+json.dumps(event, ensure_ascii=False), flush=True)
        with path.open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        display = getattr(selector, 'visualization', None)
        if display is not None and event['event'] == 'command':
            display.safe('log_book_target', dict(action='plug_'+event['stage'], arm='right',
                target_position=event['target_mm'], target_rotation=orientation.as_euler('xyz', degrees=True).tolist()))

    def checkpoint():
        selector.checkpoint()
        getattr(robot, 'operator_checkpoint', lambda: None)()

    try:
        start, orientation = _fresh_right_pose(robot)
        if selector.reset_x_mm > start[0]+3.:
            raise ValueError('first_reset初始X在抓取点前方，不能作为−X抽回目标')
        raised = start.copy(); raised[2] += config['transfer_lift_mm']
        log(dict(event='start', parameters=config, first_reset_x_mm=selector.reset_x_mm,
                 grasp_xyz_mm=start.tolist(), transfer_z_mm=float(raised[2]),
                 placement_z_mm=float(start[2]), slot_camera='head_rgbd'))
        if config['transfer_lift_mm'] > 0:
            move_axis(robot, 2, raised[2], orientation, config, checkpoint, log, 'lift', start)
        move_axis(robot, 0, selector.reset_x_mm, orientation, config, checkpoint, log, 'extract', raised)
        slot = selector.find_slot(robot, config, checkpoint, log, height_mm=float(raised[2]))
        fixed = np.array([selector.reset_x_mm, start[1], raised[2]])
        move_axis(robot, 1, slot['y_mm'], orientation, config, checkpoint, log, 'shift', fixed)
        fixed[1] = slot['y_mm']
        move_axis(robot, 0, slot['mouth_x_mm'], orientation, config, checkpoint, log, 'mouth', fixed)
        # This is explicitly relative to the measured mouth, NOT its back wall
        # and NOT whatever tracking offset remains after the mouth waypoint.
        if config['insertion_mm'] <= 0:
            raise ValueError('插入量为0：已到槽口并保持夹持；配置正插入量后重试，不自动松爪')
        move_axis(robot, 0, slot['insertion_x_mm'], orientation, config, checkpoint, log, 'insert', fixed)
        fixed[0] = slot['insertion_x_mm']
        if config['transfer_lift_mm'] > 0:
            # Use the original grasp Z, not a fresh noisy Z minus the lift.
            # Lowering is permitted only after the final insertion X arrives.
            move_axis(robot, 2, start[2], orientation, config, checkpoint, log, 'lower', fixed)
        fixed[2] = start[2]
        checkpoint()
        log(dict(event='release_requested', insertion_confirmed=True, placement_confirmed=True,
                 placement_z_mm=float(start[2])))
        open_task = dict(task, action='open_gripper', description='插槽最终X到位并竖直放下后松爪')
        if execute(robot, open_task, task_index, types, **kwargs)[1] is not True:
            raise ValueError('插槽后松爪未确认')
        # The existing open_gripper task already waits for gripper settling.
        move_axis(robot, 0, selector.reset_x_mm, orientation, config, checkpoint, log, 'withdraw_empty', fixed)
        checkpoint()
        if robot.control_joint_position(left_pos=False, right_pos=True, action='first_reset') is not True:
            raise ValueError('插槽完成后的first_reset未确认')
        log(dict(event='complete'))
        return task.get('object'), True
    except BaseException as error:
        log(dict(event='stopped', reason=f'{type(error).__name__}: {error}',
                 policy='no_automatic_release_or_reset_on_failure'))
        raise


@contextmanager
def installed(selector):
    """Only --plug replaces the post-grasp task builder and installs actions."""
    import cart_book_grasp_v1_3_motion as motion
    import cart_book_grasp_v1_3_visualization as display
    import cart_book_grasp_v2_visualization as numbered_display
    saved_previews = display.preview_targets, numbered_display.preview_targets
    saved_builder = motion._keep_cart_plus_grasp_only
    saved_install = motion._install_right_arm_reset
    wrapped = []

    def install():
        saved_install()
        import step5_execute as step5
        original = step5.execute_single_task

        def execute(robot, task, *args, **kwargs):
            if task.get('action') == 'book_plug_transfer' and task.get('book_grasp_mode') == 'cart+':
                return execute_transfer(selector, robot, task, *args, execute=original, **kwargs)
            return original(robot, task, *args, **kwargs)
        wrapped.append((step5, original))
        step5.execute_single_task = execute
    motion._keep_cart_plus_grasp_only = transfer_tasks
    motion._install_right_arm_reset = install
    display.preview_targets = numbered_display.preview_targets = preview_targets
    try:
        yield
    finally:
        motion._keep_cart_plus_grasp_only = saved_builder
        motion._install_right_arm_reset = saved_install
        display.preview_targets, numbered_display.preview_targets = saved_previews
        for step5, original in reversed(wrapped):
            step5.execute_single_task = original
