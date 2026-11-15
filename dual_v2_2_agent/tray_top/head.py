"""Decode the existing head RGB-D protocol and select a measured top-book mask."""

from dataclasses import dataclass
import io
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

import numpy as np

from .observation import EvidenceError, fit_top_surface


MAX_FRAME_BYTES = 64 * 1024 * 1024


def rigid_transform(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-5):
        raise EvidenceError('Invalid rigid transform')
    return transform


@dataclass(frozen=True)
class HeadFrame:
    rgb: np.ndarray
    depth_mm: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    base_from_camera_m: np.ndarray
    token: str
    captured_unix_s: float
    serial: str


def decode_frame(blob, *, expected_serial, max_age_s=None):
    if len(blob) > MAX_FRAME_BYTES:
        raise EvidenceError('Oversized head RGB-D payload')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        header = json.loads(str(archive['header'].item()))
        bgr = archive['native_bgr']
        depth = archive['native_depth']
    metadata = header['metadata']
    captured = float(header['captured_unix_s'])
    token = header['token']
    if (header['protocol'] != 'head-rgbd-v1' or header['depth_units'] != 'mm'
            or header['depth_encoding'] != '32FC1' or header['color_encoding'] != 'bgr8'
            or metadata['serial'] != expected_serial or metadata['native_is_maximum'] is not True
            or not isinstance(token, str) or not token or not np.isfinite(captured) or captured <= 0):
        raise EvidenceError('Head RGB-D identity, timestamp or format mismatch')
    if max_age_s is not None and not 0 <= time.time() - captured <= max_age_s:
        raise EvidenceError('Head RGB-D frame is stale')
    w, h = int(metadata['native_width']), int(metadata['native_height'])
    width, height = int(metadata['width']), int(metadata['height'])
    if (min(w, h, width, height) <= 0 or depth.shape != (h, w) or depth.dtype != np.float32
            or bgr.shape != (h, w, 3) or bgr.dtype != np.uint8):
        raise EvidenceError('Head RGB-D dimensions or array types do not match')
    sx, sy = w / width, h / height
    scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    intrinsic = scale @ np.asarray(metadata['K'], dtype=float)
    distortion = np.asarray(metadata['D'], dtype=float).reshape(-1)
    transform = rigid_transform(metadata['T_dummy_world_camera_mm']).copy()
    transform[:3, 3] /= 1000
    return HeadFrame(bgr[..., ::-1].copy(), depth, intrinsic, distortion, transform,
                     token, captured, expected_serial)


def load_frame(path, *, expected_serial):
    with Path(path).open('rb') as stream:
        return decode_frame(stream.read(MAX_FRAME_BYTES + 1), expected_serial=expected_serial)


def capture_head(*, expected_serial, save_to=None, after=None):
    # Existing board camera owner, with no robot API or camera reconfiguration.
    opener = build_opener(ProxyHandler({}))
    url = 'http://127.0.0.1:8792/frame'
    if after is not None:
        url += '?' + urlencode({'after': after})
    with opener.open(url, timeout=4) as response:
        blob = response.read(MAX_FRAME_BYTES + 1)
    frame = decode_frame(blob, expected_serial=expected_serial, max_age_s=.5)
    if frame.token == after:
        raise EvidenceError('Head RGB-D source repeated the requested frame')
    if save_to is not None:
        with Path(save_to).open('xb') as stream:
            stream.write(blob)
    return frame


def select_top(frame, masks, *, tray_roi_px, base_from_tray_m, inlier_m, level_deg,
               height_separation_m):
    """Use the highest measured, approximately level candidate inside the tray ROI."""
    x0, y0, x1, y1 = tray_roi_px
    h, w = frame.depth_mm.shape
    if (not all(type(v) is int for v in tray_roi_px)
            or not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h):
        raise EvidenceError('Tray ROI must be inside the native head image')
    if not (0 < level_deg < 90 and height_separation_m > 0):
        raise EvidenceError('Invalid top-plane tolerances')
    tray_from_camera = np.linalg.inv(rigid_transform(base_from_tray_m)) @ frame.base_from_camera_m
    roi = np.zeros((h, w), dtype=bool)
    roi[y0:y1, x0:x1] = True
    candidates = []
    for raw_mask in masks:
        mask = np.asarray(raw_mask)
        if mask.shape != (h, w) or mask.dtype != np.bool_:
            raise EvidenceError('Book mask must be a native-size boolean array')
        if not mask.any() or np.any(mask & ~roi):
            continue
        # Identical detections do not represent two physical books.
        if any(np.array_equal(mask, prior_mask) for prior_mask, _ in candidates):
            continue
        try:
            top = fit_top_surface(frame.depth_mm, frame.intrinsic, tray_from_camera,
                                  mask, inlier_m=inlier_m, distortion=frame.distortion)
        except EvidenceError:
            continue
        if abs(float(top.normal[2])) >= np.cos(np.radians(level_deg)):
            candidates.append((mask, top))
    if not candidates:
        raise EvidenceError('No measured top-book plane in the front tray ROI')
    candidates.sort(key=lambda candidate: float(candidate[1].center_m[2]), reverse=True)
    mask, top = candidates[0]
    if len(candidates) > 1 and top.center_m[2] - candidates[1][1].center_m[2] <= height_separation_m:
        raise EvidenceError('The highest book is ambiguous')
    return mask, top, tuple(other for other, _ in candidates[1:])
