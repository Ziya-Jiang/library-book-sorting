"""One head observation, then one right-to-left pass through the books.

The snapshot is deliberately reused, not represented as a fresh observation.
Wrist servo and robot feedback remain live through the existing V3.4 runtime.
"""
from copy import deepcopy
import json
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from cart_book_grasp_v2_1_numbered import NumberedCartLoop, catalog_summary, numbered_overlay
from cart_book_grasp_v2_1_selection import SelectionError, check_ready_pose, load_config
from cart_book_grasp_v2_pause import PauseRequested
from cart_book_grasp_v3_4_gpt_numbered import GPTNumberedBookSelectionAgent
from cart_book_grasp_v3_4_numbered_timing import NumberingClock
from cart_book_grasp_v3_2_head_fallback import load_config as model_config


class BatchBookSelectionAgent(GPTNumberedBookSelectionAgent):
    label = 'GPT-6 单次感知顺序抓放'
    direction_label = 'right to left'

    def ordered_numbers(self, items):
        return [item['number'] for item in reversed(items)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot = None
        self.batch_items = None
        self.batch_order = []
        self.completed_numbers = []
        self.batch_index = 0
        self.batch_output = None
        self.batch_status = 'new'

    @property
    def finished(self):
        return self.batch_items is not None and self.batch_index >= len(self.batch_order)

    def attach_camera(self, camera):
        """Keep downstream masks, point clouds and visualization on the same RGB-D."""
        self.observation_camera = camera
        live_capture = camera.capture_image
        self.live_capture = live_capture  # Slot observations must bypass the batch snapshot.

        def capture():
            if self.snapshot is None:
                return live_capture()
            source = self.snapshot
            for key, value in source['camera'].items():
                setattr(camera, key, deepcopy(value))
            camera.rgb_image = source['rgb'].copy()
            camera.rgb_right_image = source['rgb'].copy()
            camera.depth_map = source['depth'].copy()
            camera.batch_source_captured_monotonic = source['captured_monotonic']
            return camera.rgb_image, camera.depth_map

        camera.capture_image = capture
        return camera

    def _write(self):
        if self.batch_output is None:
            return
        self.last_decision.update(batch_status=self.batch_status,
            order_right_to_left=self.batch_order, completed_numbers=list(self.completed_numbers),
            next_index=self.batch_index, head_reobservation=False, scene_comparison=False)
        payload = json.dumps(self.last_decision, ensure_ascii=False, indent=2)
        (self.batch_output/'decision.json').write_text(payload)
        (self.batch_output/'batch.json').write_text(payload)

    def complete_current(self):
        if self.finished or self.batch_items is None:
            raise SelectionError('没有可确认完成的批次目标')
        number = self.batch_order[self.batch_index]
        if self.last_decision.get('selected_number') != number:
            raise SelectionError('当前执行目标与批次顺序不一致')
        self.completed_numbers.append(number)
        self.batch_index += 1
        self.batch_status = 'complete' if self.finished else 'ready'
        self._write()

    def end_batch(self, reason):
        if not self.finished:
            self.batch_status = 'stopped'
            self.last_decision['stop_reason'] = reason
        self._write()

    def restart_batch(self):
        """Start a new observation only after the operator requests a retry."""
        self.last_decision['operator_requested_retry'] = True
        self._write()
        self.clear_batch_snapshot()

    def clear_batch_snapshot(self):
        """Discard one completed/aborted pass without modifying physical state."""
        self._reset()
        self.snapshot = None
        self.batch_items = None
        self.batch_order = []
        self.completed_numbers = []
        self.batch_index = 0
        self.batch_output = None
        self.batch_status = 'new'
        camera = getattr(self, 'observation_camera', None)
        if camera is not None and hasattr(camera, 'batch_source_captured_monotonic'):
            del camera.batch_source_captured_monotonic

    def _overlay(self, selected_number=None):
        image = numbered_overlay(self.snapshot['rgb'], self.batch_items, selected_number)
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (25, 25, 25), -1)
        order = ' -> '.join(map(str, self.batch_order))
        cv2.putText(image, 'FIRST HEAD FRAME | '+self.direction_label+': '+order, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def select_instances(self, rgb, depth, intrinsic, c2w, book_geom, prepare_geometry,
                         on_recognized=None, *, head_captured_at=None, dual_config=None,
                         observation_guard=lambda: None):
        config = load_config(self.config_path, require_semantic_model=False)

        def checkpoint():
            self.checkpoint()
            observation_guard()
            check_ready_pose(self.robot, config)
            self.check_feedback()

        try:
            checkpoint()
            if self.batch_items is None:
                if self.batch_output is not None:
                    raise SelectionError('本批首次识别已失败；等待回车复位并重新识别，不自动重复检测')
                if head_captured_at is None or not np.isfinite(head_captured_at):
                    raise SelectionError('缺少首次头部采集时间')
                clock = NumberingClock(head_captured_at, config['max_observation_age_seconds'],
                                       model_config()['timeout_seconds'])
                clock.check()
                scene_id = uuid.uuid4().hex
                self.batch_output = self.output_root/scene_id
                self.batch_output.mkdir(parents=True, exist_ok=False)
                self.last_decision = dict(scene_id=scene_id, status='observing',
                    selection_mode='batch_right_to_left', output=str(self.batch_output),
                    original_captured_monotonic=head_captured_at,
                    source_policy='reuse_first_head_rgbd_and_geometry_for_whole_batch')
                camera = getattr(self, 'observation_camera', None)
                self.snapshot = dict(rgb=np.array(rgb, copy=True), depth=np.array(depth, copy=True),
                    K=np.array(intrinsic, copy=True), c2w=np.array(c2w, copy=True),
                    captured_monotonic=head_captured_at,
                    camera={key: deepcopy(getattr(camera, key)) for key in
                        ('K', 'D', 'R', 'T', 'w', 'h', 'captured_unix_s', 'frame_token')
                        if camera is not None and hasattr(camera, key)})
                Image.fromarray(rgb).save(self.batch_output/'head_rgb.png')
                np.save(self.batch_output/'depth.npy', depth)
                (self.batch_output/'camera.json').write_text(json.dumps(dict(
                    K=self.snapshot['K'].tolist(), c2w=self.snapshot['c2w'].tolist(),
                    original_captured_monotonic=head_captured_at), indent=2))
                timings = {}
                clock.enter('catalog')

                def catalog_checkpoint():
                    checkpoint()
                    clock.check()

                self.batch_items = self.perceive_catalog(rgb, depth, intrinsic, c2w, book_geom,
                    config, self.batch_output, catalog_checkpoint, timings)
                self.batch_order = self.ordered_numbers(self.batch_items)
                self.last_candidates = self.batch_items
                self.last_decision.update(timings=timings, catalog=catalog_summary(self.batch_items))
                # Metric targets for every book use this same original frame.
                # A bad target is retained at its original position, never silently skipped.
                for item in self.batch_items:
                    checkpoint()
                    if item.get('selection_blocked'):
                        continue
                    try:
                        prepare_geometry(item)
                        if item.get('geometry') is None:
                            raise ValueError('首次头部目标几何未完成')
                    except PauseRequested:
                        raise
                    except (ValueError, np.linalg.LinAlgError) as error:
                        item.update(geometry=None, geometry_status='failed', error=str(error))
                    checkpoint()
                self.batch_status = 'ready'
                self.last_decision['geometry_results'] = [dict(number=item['number'],
                    status=item['geometry_status'], error=item.get('error', ''),
                    selection_blocked=item.get('selection_blocked', '')) for item in self.batch_items]
                print('首次头部识别完成；'+self.direction_label+'，依次处理：'+
                      ' → '.join(map(str, self.batch_order))+'；后续复用首次信息，不重新检测或比较深度。', flush=True)
                Image.fromarray(self._overlay()).save(self.batch_output/'head_numbered.png')
            if self.finished:
                raise SelectionError('本批所有书已完成，不重复抓取')
            number = self.batch_order[self.batch_index]
            item = next(item for item in self.batch_items if item['number'] == number)
            self.last_decision.update(status='selected', selected_id=item['id'],
                selected_number=item['number'],
                original_frame_age_seconds=time.monotonic()-self.snapshot['captured_monotonic'])
            self.last_image = self._overlay(item['number'])
            self.last_gate_image = self.last_image
            self.last_review_image = self.last_image
            Image.fromarray(self.last_image).save(self.batch_output/f"head_selected_{item['number']:02d}.png")
            if item.get('selection_blocked') or item.get('geometry') is None:
                raise SelectionError(f"编号 {item['number']} 无有效首次目标："+
                                     (item.get('selection_blocked') or item.get('error', '几何缺失')))
            if self.observe_only:
                raise SelectionError('仅观测批次完成；未执行抓取')
            checkpoint()
            self.batch_status = 'executing'
            print(f"本批 {self.batch_index+1}/{len(self.batch_order)}：抓放编号 {item['number']}，使用首次RGB-D与已保存目标。", flush=True)
            # Motion code may attach per-execution metadata; never mutate cached geometry.
            return deepcopy(item)
        except BaseException as error:
            self.batch_status = 'stopped'
            self.last_decision.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self._write()
            if on_recognized is not None:
                on_recognized()


class BatchCartLoop(NumberedCartLoop):
    completion_message = '本批从右到左全部抓放完成，结束；不重新识别、不自动重复。'
    startup_message = ('V3.4 多书顺序抓放：头部GPT只识别一次，从机器人右到左逐本抓起、放下；'
                       '成功后自动下一本，全部完成结束；失败后回车复位并重新识别整批。'
                       's 停止后续，p 中断并复位后结束本批。')

    def __init__(self, selector):
        super().__init__(selector)
        self.result_pending = False
        self.batch_ended = False

    def advance_pass(self):
        return False  # Normal multi mode has exactly one pass.

    def handle_interrupt(self, error):
        handled = super().handle_interrupt(error)
        if handled:
            self.selector.end_batch('操作员p中断并执行原复位处理；本批不继续')
            self.batch_ended = True
        return handled

    def wait_for_batch_retry(self):
        print('本批暂停：当前书未完成抓放，后续书不执行；保持夹爪，不自动松爪或复位。', flush=True)
        while not self.stop.is_set():
            try:
                line = self.read_line(
                    '\n按回车先执行 first_reset，再重新采图识别整批书；s 退出；p 松爪复位后退出。\n> ',
                    self.check_pause)
            except SelectionError:
                if self.stop.is_set():
                    return False
                raise
            if line:
                print('请按回车重新开始整批，或按 s 退出。', flush=True)
                continue
            self.check_selection_checkpoint()
            self.selector.restart_batch()
            # Returning cart+ runs the existing first_reset-before-perception
            # path. Never send reset/enable commands from this input handler.
            return True
        return False

    def next_task(self):
        self.armed = True
        if self.batch_ended:
            return None
        if self.result_pending:
            self.result_pending = False
            if self.last_success is not True:
                self.selector.end_batch('当前任务未确认完成；不跳过、不继续下一本')
                if not self.wait_for_batch_retry():
                    self.batch_ended = True
                    return None
            else:
                self.selector.complete_current()
        if self.stop.is_set():
            self.selector.end_batch('操作员按s结束本批')
            self.batch_ended = True
            return None
        self.check_pause()
        if self.selector.finished and not self.advance_pass():
            self.batch_ended = True
            print(self.completion_message, flush=True)
            return None
        self.cycles += 1
        self.last_success = None
        self.result_pending = True
        print('首次采集头部并识别全部书。' if self.selector.batch_items is None else
              '继续下一本，复用首次头部识别结果。', flush=True)
        return 'cart+'
