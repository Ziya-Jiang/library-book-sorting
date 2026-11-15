"""Read the active dual-gripper model and camera on 155 for a right-arm preview."""

import json
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tray_top.head import capture_head
from .right_probe import measure_top, numeric_preview


PROFILE = Path('/mnt/data/erazer/book_grasp/dual_gripper_runtime/candidate_20261111')
ORIGIN = 'http://127.0.0.1:8810'


def feedback_reader():
    info = json.loads((PROFILE / 'runtime/web.json').read_text())
    token = parse_qs(urlsplit(info['url']).fragment)['token'][0]
    opener = build_opener(ProxyHandler({}))

    def feedback():
        request = Request(ORIGIN + '/api/state', headers={'X-Arm-Token': token})
        with opener.open(request, timeout=3) as response:
            state = json.load(response)
        if (state.get('rt_assembly') != 'dual_gripper' or state.get('simulation')
                or state.get('gripper_installed') != [False, True]
                or not state.get('fresh') or not 0 <= state['age_ms'] <= 100):
            raise RuntimeError('需要当前双夹爪配置的新鲜真机反馈')
        return state

    return feedback


def load_board_model():
    sys.path.insert(0, str(PROFILE))
    try:
        from backend.board_servo.model import Model
    finally:
        sys.path.remove(str(PROFILE))
    return Model(PROFILE / 'backend/board_servo')


def observe(parameters, *, out, serial, polygon=None):
    """Measure the right arm and optionally fit an operator-selected top surface."""
    feedback = feedback_reader()
    model = load_board_model()
    before = feedback()
    out.mkdir(parents=True, exist_ok=False)
    frame = capture_head(expected_serial=serial, save_to=out / 'head.npz')
    after = feedback()
    arms = after['state']['arms']
    joints = np.asarray([arm['position'] for arm in arms]).reshape(14)
    previous = np.asarray([arm['position'] for arm in before['state']['arms']]).reshape(14)
    positions, quaternions = model.tcp(joints)
    rotation = Rotation.from_quat(quaternions[0, 1, [1, 2, 3, 0]])
    tcp_mm = positions[0, 1] * 1000
    flange_mm = tcp_mm - rotation.apply(model.offsets[1] * 1000)
    flange_pose = np.r_[flange_mm, rotation.as_euler('xyz', degrees=True)]

    report = dict(mode='live_right_arm_preview', runtime_host='155', motion_enabled=False,
                  frame_token=frame.token, captured_unix_s=frame.captured_unix_s,
                  frame_age_s=time.time() - frame.captured_unix_s, board_model_sha256=model.digest,
                  feedback_age_ms=after['age_ms'], right_tcp_mm=tcp_mm.tolist(),
                  right_flange_pose_mm_deg=flange_pose.tolist(),
                  observed_joint_change_deg=float(np.degrees(np.max(np.abs(joints - previous)))),
                  output=str(out), status='captured', execution_ready=False)
    rgb = frame.rgb.copy()
    if polygon is not None:
        pixels = np.asarray(polygon)
        h, w = frame.depth_mm.shape
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 3 <= len(pixels) <= 40
                or pixels.dtype.kind not in 'iu' or np.any(pixels < 0)
                or np.any(pixels >= [w, h])):
            raise ValueError('顶书轮廓须为图像内的 3～40 个整数 [u,v] 顶点')
        contour = pixels.astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        cv2.polylines(rgb, [contour], True, (255, 180, 0), 2)
        try:
            observation = measure_top(frame, [mask.astype(bool)], dict(
                tray_roi_px=[0, 0, w, h], base_from_tray_m=np.eye(4),
                inlier_m=.005, level_deg=12, height_separation_m=.005))
            preview = numeric_preview(observation, flange_pose, parameters)
            for stage in preview['stages']:
                stage['flange_target_mm'] = stage.pop('tcp_target_mm')
            for sample in preview['pitch_samples'] + preview['j7_only_samples']:
                sample['flange_mm'] = sample.pop('tcp_mm')
            report.update(status='geometry_preview', target_source='operator-selected top polygon',
                          rear_edge_confirmed=False, geometry=preview)
        except ValueError as error:
            report.update(status='geometry_rejected', reason=str(error))
    if not cv2.imwrite(str(out / 'head.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError('无法保存头部相机预览')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report
