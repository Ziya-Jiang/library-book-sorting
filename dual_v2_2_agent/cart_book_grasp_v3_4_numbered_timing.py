"""Phase deadlines and fresh-frame validation for slow model-assisted numbering."""
import time
import cv2
import numpy as np
from cart_book_grasp_v2_1_selection import SelectionError


class NumberingClock:
    def __init__(self, captured_at, selection_seconds, request_seconds):
        self.captured_at = captured_at
        # Catalog includes request plus bounded encoding/depth/preview work.
        self.limits = dict(capture=selection_seconds, catalog=request_seconds+10.,
                           selection=selection_seconds, verification=10.)
        self.phase = 'capture'
        self.started = captured_at
        self.history = []

    def enter(self, phase):
        self.check()
        self.phase = phase
        self.started = time.monotonic()
        self.history.append(dict(phase=phase, started_monotonic=self.started))

    def remaining(self):
        return max(0., self.limits[self.phase]-(time.monotonic()-self.started))

    def check(self):
        age = time.monotonic()-self.started
        if not np.isfinite(age) or age < 0 or age > self.limits[self.phase]:
            label = dict(capture='采集图像', catalog='GPT编号处理',
                         selection='显示后的编号选择', verification='选中后的新帧复核')[self.phase]
            raise SelectionError(f'{label}超时；按回车重新识别，不复用旧编号')

    def summary(self):
        return dict(policy='separate_model_and_selection_with_fresh_frame_check',
                    original_captured_monotonic=self.captured_at,
                    original_frame_age_seconds=time.monotonic()-self.captured_at,
                    phase=self.phase, limits_seconds=self.limits, phases=self.history)


def validate_scene(rgb, depth, fresh_rgb, fresh_depth, mask):
    """A stationary-scene consistency check, not a new semantic identity claim."""
    if fresh_rgb.shape != rgb.shape or fresh_depth.shape != depth.shape:
        raise SelectionError('新头部帧尺寸变化；请重新编号')
    mask = np.asarray(mask, bool)
    if mask.shape != depth.shape or not mask.any():
        raise SelectionError('选中编号区域无效')
    # Include the boundary: a book moving out of its old polygon is a change.
    region = cv2.dilate(mask.astype(np.uint8), np.ones((13,13), np.uint8)).astype(bool)
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    after = cv2.cvtColor(fresh_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    change = np.abs(after-before)[region]
    old = np.asarray(depth, float).copy()
    new = np.asarray(fresh_depth, float).copy()
    for array in (old, new):
        valid = array[np.isfinite(array) & (array > 0)]
        if len(valid) and np.median(valid) <= 20:array *= 1000.
    valid = mask & np.isfinite(old) & np.isfinite(new) & (old>0) & (new>0)
    support = float(valid.sum()/mask.sum())
    if valid.sum()<60 or support<.5:
        raise SelectionError('新帧缺少同区域实测深度；请重新编号')
    difference = np.abs(old[valid]-new[valid])
    report = dict(rgb_mean_change=float(change.mean()),
                  rgb_changed_fraction=float(np.mean(change>32)),
                  depth_median_change_mm=float(np.median(difference)),
                  depth_changed_fraction=float(np.mean(difference>20)), valid_fraction=support,
                  limits=dict(rgb_mean=18., rgb_fraction=.25, depth_median_mm=10., depth_fraction=.25))
    if (report['rgb_mean_change']>18 or report['rgb_changed_fraction']>.25 or
            report['depth_median_change_mm']>10 or report['depth_changed_fraction']>.25):
        raise SelectionError('所选书画面或深度在等待期间变化；请重新编号：'+str(report))
    return report
