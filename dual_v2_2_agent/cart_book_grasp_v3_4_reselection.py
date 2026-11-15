"""Opt-in, bounded new-book selection after an unexecuted coarse-Y IK failure."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import numpy as np

_active = ContextVar('book_ik_reselection', default=None)
CONFIG = Path(__file__).with_suffix('.json')


def load_config():
    c = json.loads(CONFIG.read_text())
    if type(c.get('max_retries')) is not int or not 1 <= c['max_retries'] <= 3:
        raise ValueError('GPT reselection max_retries must be 1..3')
    for key in ('max_y_offset_mm', 'max_forward_mm', 'same_failed_target_radius_mm'):
        if type(c.get(key)) not in (int, float) or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid GPT reselection setting: '+key)
    return c


def base_center(selected, c2w):
    transform = np.asarray(c2w, float).copy()
    if np.linalg.norm(transform[:3, 3]) < 10:
        transform[:3, 3] *= 1000.
    return transform[:3, :3] @ (np.asarray(selected['point_camera_m']) * 1000.) + transform[:3, 3]


def distance_report(selected, c2w, pose, parameters, config):
    from cart_book_grasp_v1_3_depth import tool_rotation
    if pose is None:
        raise ValueError('No measured right TCP for GPT distance review')
    xyz, angles = map(lambda value: np.asarray(value, float), pose)
    center = base_center(selected, c2w)
    finger_x = -tool_rotation(angles)[0, 2] * parameters['fingertip_offset_mm']
    target_x = selected['book_surface_x_mm'] + parameters['grasp_insertion_mm'] - finger_x
    dx, dy = float(target_x - xyz[0]), float(center[1] - xyz[1])
    if not np.isfinite([dx, dy, *center]).all():
        raise ValueError('Non-finite selected-book distance')
    reasons = []
    if abs(dy) > config['max_y_offset_mm']: reasons.append('Y offset too far')
    if dx > config['max_forward_mm']: reasons.append('forward endpoint too far')
    if dx <= 0: reasons.append('forward endpoint not ahead of TCP')
    return dict(accepted=not reasons, reasons=reasons, current_tcp_mm=xyz.tolist(),
                center_base_mm=center.tolist(), y_offset_mm=dy, forward_mm=dx,
                max_y_offset_mm=config['max_y_offset_mm'], max_forward_mm=config['max_forward_mm'])


def current():
    return _active.get()


@contextmanager
def enabled():
    policy = Reselection()
    token = _active.set(policy)
    try:
        yield policy
    finally:
        _active.reset(token)


class Reselection:
    def __init__(self, max_retries=None):
        self.config = load_config()
        self.max_retries = self.config['max_retries'] if max_retries is None else max_retries
        self.reset()

    def reset(self):
        self.attempts = 0
        self.pending = False
        self.context = None
        self.selected = None
        self.rejected = []

    def capture(self, selected, c2w):
        center = base_center(selected, c2w)
        yy, xx = np.nonzero(selected['mask'])
        h, w = selected['mask'].shape
        self.selected = dict(center_base_mm=center.tolist(), rejected_box_0_1000=[
            float(xx.min()) / (w-1)*1000, float(yy.min()) / (h-1)*1000,
            float(xx.max()) / (w-1)*1000, float(yy.max()) / (h-1)*1000])

    def on_result(self, task, result, failure):
        if (result or task.get('action') != 'move_above_grasp'
                or task.get('arm') != 'right' or task.get('book_grasp_mode') != 'cart+'
                or not failure or failure.get('status') != 7
                or failure.get('command_matched') is not True
                or failure.get('execution_started') is not False
                or failure.get('joint_command') is not False or self.selected is None):
            return
        if self.attempts >= self.max_retries:
            print('V3.4 GPT 重选次数已耗尽；保持暂停，等待人工检查。', flush=True)
            return
        self.attempts += 1
        self.rejected.append(dict(self.selected))
        self.context = dict(mode='reselect_after_ik_failure', attempt=self.attempts,
            reason='coarse_Y_alignment_IK_FAIL_before_execution',
            rejected_candidates=list(self.rejected),
            failed_target_mm=task.get('target_position'),
            failed_orientation_deg=task.get('target_rotation'))
        self.pending = True
        print(f'V3.4 粗 Y 对准 IK 无解，取消旧任务；准备 GPT-6 重选 '
              f'({self.attempts}/{self.max_retries})。', flush=True)

    def take_retry(self):
        retry = self.pending
        self.pending = False
        return retry

    def selection_context(self):
        return self.context

    def review(self, selected, book_mask, rgb, depth, intrinsic, c2w, **kwargs):
        self.config = load_config()  # Applied at perception boundary, not mid-motion.
        self.max_retries = self.config['max_retries']
        robot = kwargs.get('observation_robot')
        if robot is not None:
            selected, book_mask = self.review_distance(
                selected, book_mask, rgb, depth, intrinsic, c2w, robot,
                box_detector=kwargs.get('box_detector'))
        self.capture(selected, c2w)
        if self.context is not None:
            center = np.asarray(self.selected['center_base_mm'])
            radius = self.config['same_failed_target_radius_mm']
            if any(np.linalg.norm(center-np.asarray(old['center_base_mm'])) < radius
                   for old in self.rejected):
                raise ValueError(f'GPT 重选仍指向已失败目标附近（{radius:g} mm）；取消自动重试')
            self.context = None
        return selected, book_mask

    def review_distance(self, selected, book_mask, rgb, depth, intrinsic, c2w, robot,
                        *, box_detector=None, pose_reader=None, parameters=None,
                        recover_fn=None, center_fn=None):
        from cart_book_grasp_v3_2_head_fallback import recover, center_from_mask, direct_book_mask
        from cart_book_grasp_v1_3_depth import whole_book_depth
        if pose_reader is None:
            import step5_execute as step5
            from cart_book_grasp_v1_3_motion import _right_tool_mm_euler
            pose_reader = lambda: _right_tool_mm_euler(robot, step5)
        if parameters is None:
            from cart_book_grasp_v1_3_motion import _load_cart_plus_parameters
            parameters = _load_cart_plus_parameters()
        before = distance_report(selected, c2w, pose_reader(), parameters, self.config)
        if before['accepted']:
            return selected, book_mask
        if self.attempts >= self.max_retries:
            raise ValueError('目标仍过远，GPT 重选预算已用完；不发送粗定位目标')
        self.capture(selected, c2w)
        self.rejected.append(dict(self.selected))
        self.attempts += 1
        context = dict(mode='reselect_near_book', reason='selected_book_outside_coarse_travel',
            attempt=self.attempts, distance_report=before, rejected_candidates=list(self.rejected),
            rejected_box_0_1000=self.selected['rejected_box_0_1000'])
        print('V3.4 目标距离过远，先交 GPT-6 重选：'+json.dumps(before, ensure_ascii=False), flush=True)
        checkpoint = getattr(robot, 'operator_checkpoint', lambda: None)
        checkpoint()
        detections, books = (recover_fn or recover)(rgb, box_detector,
            checkpoint=checkpoint, review_context=context)
        if len(detections) != 1:
            raise ValueError('GPT distance review did not select one book')
        score, mask, _ = detections[0]
        candidate = (center_fn or center_from_mask)(mask, depth, intrinsic)
        full = direct_book_mask(books, candidate['mask'].shape)
        candidate.update(whole_book_depth(full, depth, intrinsic, c2w))
        candidate.update(score=float(score), selection_source='gpt6_polygon_fallback')
        checkpoint()
        after = distance_report(candidate, c2w, pose_reader(), parameters, self.config)
        if not after['accepted']:
            raise ValueError('GPT 重选后实测目标距离仍不合适：'+json.dumps(after, ensure_ascii=False))
        candidate['distance_reselection'] = dict(before=before, after=after)
        print('V3.4 GPT 重选后实测距离通过：'+json.dumps(after, ensure_ascii=False), flush=True)
        return candidate, full
