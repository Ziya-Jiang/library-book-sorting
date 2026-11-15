"""Measure a selected top book and build right-arm waypoints on 155."""

import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tray_top.head import capture_head
from tray_top.observation import EvidenceError, fit_top_surface
from .right_probe_live import feedback_reader, load_board_model


FRAME_COUNT = 5
DEPTH_AGREEMENT_MM = 5.0
PLANE_TOLERANCE_M = .005
FRAME_WINDOW_S = 1.5


def top_mask(shape, polygon):
    height, width = shape
    points = np.asarray(polygon)
    if (points.shape != (4, 2) or points.dtype.kind not in 'iu'
            or np.any(points < 0) or np.any(points >= [width, height])):
        raise EvidenceError('顶书选区需要原图内的四个 [u,v] 角点，按轮廓顺序排列')
    contour = points.astype(np.int32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 200:
        raise EvidenceError('顶书选区必须是有面积的凸四边形')
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    return mask.astype(bool)


def stable_depth(frames, mask, diagnostics, *, require_spatial_coverage=True):
    """Fuse measured pixels that agree across a majority of captures."""
    first = frames[0]
    last = frames[-1]
    tokens = [frame.token for frame in frames]
    stamps = np.asarray([frame.captured_unix_s for frame in frames])
    if (len(frames) != FRAME_COUNT or len(set(tokens)) != FRAME_COUNT
            or len({token.rsplit(':', 1)[0] for token in tokens}) != 1
            or np.any(np.diff(stamps) <= 0) or stamps[-1] - stamps[0] > FRAME_WINDOW_S):
        raise EvidenceError('RGB-D 取帧重复、顺序错误或跨度过长')
    for frame in frames:
        if (frame.serial != first.serial or frame.depth_mm.shape != mask.shape
                or not np.array_equal(frame.intrinsic, first.intrinsic)
                or not np.array_equal(frame.distortion, first.distortion)
                or not np.array_equal(frame.base_from_camera_m, first.base_from_camera_m)):
            raise EvidenceError('测量期间相机或标定变更')

    depth = np.stack([frame.depth_mm[mask] for frame in frames])
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 5000)
    # An invalid sample cannot vote for the median. Use the middle valid order
    # statistic, then require a majority of all captures within 5 mm of it.
    sorted_depth = np.sort(np.where(valid, depth, np.inf), axis=0)
    median_indices = np.maximum((valid.sum(axis=0) - 1) // 2, 0)
    median = sorted_depth[median_indices, np.arange(depth.shape[1])]
    finite_median = np.where(np.isfinite(median), median, 0)
    consistent = valid & (np.abs(np.where(valid, depth, 0) - finite_median) <= DEPTH_AGREEMENT_MM)
    stable = consistent.sum(axis=0) >= FRAME_COUNT // 2 + 1
    stable_mask = np.zeros(mask.shape, bool)
    stable_mask[mask] = stable
    diagnostics.update(frame_tokens=tokens, capture_span_s=float(stamps[-1] - stamps[0]),
                       mask_pixels=int(mask.sum()), stable_depth_pixels=int(stable.sum()),
                       stable_depth_fraction=float(stable.mean()),
                       depth_agreement_mm=DEPTH_AGREEMENT_MM)

    rows, cols = np.nonzero(mask)
    # Support from only one end of the cover cannot establish its far edge.
    halves = (rows <= np.median(rows), rows > np.median(rows),
              cols <= np.median(cols), cols > np.median(cols))
    coverage = [float(stable[half].mean()) for half in halves]
    diagnostics['stable_fraction_by_image_half'] = coverage
    diagnostics['require_spatial_coverage'] = require_spatial_coverage
    if require_spatial_coverage and min(coverage) < .4:
        raise EvidenceError('顶书选区一端缺少重复有效深度')
    fused = np.zeros(mask.shape, np.float32)
    fused[stable_mask] = median[stable]
    return fused, stable_mask


def measure(frames, mask, diagnostics):
    """Fit the top plane for the raise, advance and pitch stages."""
    last = frames[-1]
    fused, stable_mask = stable_depth(frames, mask, diagnostics)
    plane = fit_top_surface(fused, last.intrinsic, last.base_from_camera_m, stable_mask,
                            inlier_m=PLANE_TOLERANCE_M, distortion=last.distortion)
    points = plane.points_m * 1000
    diagnostics.update(plane_inlier_fraction=plane.inlier_fraction,
                       plane_max_residual_mm=plane.residual_m * 1000,
                       plane_tilt_deg=float(np.degrees(np.arccos(np.clip(plane.normal[2], -1, 1)))),
                       supported_points=len(points))
    return points


def approach_target(frames, mask, diagnostics, position_mm, quaternion_wxyz, *, require_spatial_coverage=True):
    """Use measured base Y within the selected top cover for lateral alignment."""
    fused, stable_mask = stable_depth(frames, mask, diagnostics, require_spatial_coverage=require_spatial_coverage)
    if stable_mask.sum() < max(30, .4 * mask.sum() if require_spatial_coverage else 0):
        raise EvidenceError('顶书选区内有效重复深度不足，无法定位中心Y')
    frame = frames[-1]
    v, u = np.nonzero(stable_mask)
    pixels = np.column_stack((u, v)).astype(float)
    rays = cv2.undistortPoints(pixels[:, None], frame.intrinsic, frame.distortion)[:, 0]
    camera_mm = np.column_stack((rays, np.ones(len(rays)))) * fused[v, u, None]
    transform = frame.base_from_camera_m
    points_mm = camera_mm @ transform[:3, :3].T + transform[:3, 3] * 1000
    if not np.isfinite(points_mm).all():
        raise EvidenceError('顶书中心的深度反投影无效')
    center = np.median(points_mm, axis=0)
    target = np.asarray(position_mm, dtype=float).copy()
    target[1] = center[1]
    return dict(
        mode='approach_only',
        measured=dict(method='median_measured_base_y_in_selected_top_cover',
                      center_mm=center.tolist(), observed_min_mm=points_mm.min(axis=0).tolist(),
                      observed_max_mm=points_mm.max(axis=0).tolist()),
        offset_y_mm=float(target[1] - position_mm[1]),
        stages=[dict(stage='approach_y', translation_axis='Y',
                     tcp_m_wxyz=np.r_[target * .001, quaternion_wxyz].tolist())],
    )


def waypoints(points, position_mm, orientation, offset_mm, parameters):
    """Use a TCP path for translation and a fixed flange for the pitch arc."""
    points = np.asarray(points, float)
    position = np.asarray(position_mm, float)
    offset = np.asarray(offset_mm, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        raise EvidenceError('没有足够的实测顶面点')
    low = points.min(axis=0)
    high = points.max(axis=0)
    center_y = float(np.median(points[:, 1]))
    upper_z = float(high[2])
    far_x = float(high[0])
    clearance = parameters['raise_clearance_mm']
    beyond = parameters['back_edge_overrun_mm']

    align = position.copy()
    align[1] = center_y
    raised = align.copy()
    raised[2] = max(align[2], upper_z + clearance)
    advance = raised.copy()
    advance[0] = max(raised[0], far_x + beyond)
    tool_direction = orientation.apply(offset / np.linalg.norm(offset))
    # Preserve tool roll while aiming its forward axis along the push direction.
    correction, _ = Rotation.align_vectors([[1., 0., 0.]], [tool_direction])
    forward = correction * orientation

    def pose(xyz, rotation):
        return np.r_[xyz * .001, rotation.as_quat()[[3, 0, 1, 2]]].tolist()

    stages = [dict(stage='approach_y', tcp_m_wxyz=pose(align, orientation), translation_axis='Y'),
              dict(stage='raise_z', tcp_m_wxyz=pose(raised, orientation), translation_axis='Z'),
              dict(stage='aim_forward', tcp_m_wxyz=pose(raised, forward), translation_axis=None),
              dict(stage='advance_x', tcp_m_wxyz=pose(advance, forward), translation_axis='X')]
    flange = advance - forward.apply(offset)
    pitch_deg = parameters['wrist_pitch_down_deg']
    steps = int(np.ceil(pitch_deg / parameters['wrist_preview_step_deg']))
    pitch = []
    for angle in np.linspace(0., pitch_deg, steps + 1):
        rotation = Rotation.from_euler('y', angle, degrees=True) * forward
        point = flange + rotation.apply(offset)
        pitch.append(dict(angle_deg=float(angle), tcp_m_wxyz=pose(point, rotation)))
    end = np.asarray(pitch[-1]['tcp_m_wxyz'][:3]) * 1000
    return dict(measured=dict(center_y_mm=center_y, median_top_z_mm=float(np.median(points[:, 2])),
                              highest_top_z_mm=upper_z, observed_max_x_mm=far_x,
                              observed_min_mm=low.tolist(), observed_max_mm=high.tolist(),
                              rear_edge_confirmed=False),
                stages=stages, pitch_arc=pitch,
                pitch_geometry=dict(flange_fixed_mm=flange.tolist(),
                                    aim_rotation_deg=float(np.degrees(correction.magnitude())),
                                    advance_clearance_z_mm=float(advance[2] - upper_z),
                                    final_x_minus_observed_rear_mm=float(end[0] - far_x),
                                    final_z_minus_highest_top_mm=float(end[2] - upper_z),
                                    contact_confirmed=False))


def acquire_trial(parameters, *, out, serial, polygon=None, approach_only=False):
    """Run the first 155 trial from settled feedback and measured RGB-D."""
    from .right_probe_live import PROFILE

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='rejected', runtime_host='155', motion_enabled=False, execution_ready=False,
                  output=str(out), depth={})
    try:
        feedback = feedback_reader()
        model = load_board_model()
        before = feedback()
        frames = []
        for i in range(FRAME_COUNT):
            save = out / f'head_{i:02d}.npz' if i in (0, FRAME_COUNT - 1) else None
            frames.append(capture_head(expected_serial=serial, save_to=save,
                                       after=frames[-1].token if frames else None))
        last = frames[-1]
        if polygon is None:
            from .right_trial_gpt import infer_corners
            report['selection'] = {}
            polygon = infer_corners(last, report['selection'])
        else:
            report['selection'] = dict(source='manual', frame_token=last.token, corners_px=polygon)
        mask = top_mask(last.depth_mm.shape, polygon)
        after = feedback()
        overlay = cv2.cvtColor(last.rgb, cv2.COLOR_RGB2BGR)
        cv2.polylines(overlay, [np.asarray(polygon, np.int32)], True, (0, 180, 255), 2)
        if not cv2.imwrite(str(out / 'top_book.jpg'), overlay):
            raise OSError('无法保存顶书选区预览')
        joints = np.asarray([arm['position'] for arm in after['state']['arms']]).reshape(14)
        old = np.asarray([arm['position'] for arm in before['state']['arms']]).reshape(14)
        velocity = np.asarray([arm['velocity'] for arm in after['state']['arms']]).reshape(14)
        if (not np.isfinite(np.r_[joints, old, velocity]).all()
                or np.max(np.abs(velocity)) > .02 or np.max(np.abs(joints - old)) > np.deg2rad(.2)):
            raise EvidenceError('取帧期间机械臂未保持静止，请停稳后重取')
        feedback_limits = np.asarray([arm['soft_position'] for arm in after['state']['arms']]).reshape(14, 2)
        feedback_limits = np.stack([np.maximum(feedback_limits[:, 0], model.bounds[0]),
                                    np.minimum(feedback_limits[:, 1], model.bounds[1])])
        velocity_limits = np.asarray([arm['soft_velocity'] for arm in after['state']['arms']]).reshape(14, 2)
        velocity_caps = np.minimum(-velocity_limits[:, 0], velocity_limits[:, 1]) * .1
        if not np.isfinite(velocity_caps).all() or np.any(velocity_caps <= 0):
            raise EvidenceError('无有效的实机速度限位')
        grip = [float(gripper['position'][0]) for gripper in after['state']['grippers']]
        if len(grip) != 2 or grip[0] != 0 or not 0 <= grip[1] <= 1:
            raise EvidenceError('双夹爪反馈与当前装配不一致')
        actual_tcp, quaternions = model.tcp(joints)
        position = actual_tcp[0, 1] * 1000
        rotation = Rotation.from_quat(quaternions[0, 1, [1, 2, 3, 0]])
        offset = model.offsets[1] * 1000
        if not np.allclose(offset, parameters['nominal_contact_offset_tool_mm']):
            raise EvidenceError('当前模型 TCP 偏置与拨书预览标定不一致')
        profile = json.loads((PROFILE / 'profile_manifest.json').read_text())
        report.update(captured_unix_s=last.captured_unix_s, frame_token=last.token,
                      observation_age_at_feedback_s=time.time() - last.captured_unix_s,
                      feedback_age_ms=after['age_ms'], board_model_sha256=model.digest,
                      urdf_sha256=profile['files']['configuration/dual_gripper.urdf'],
                      start_joints_rad=joints.tolist(), arm_limits_rad=feedback_limits.tolist(),
                      planner_velocity_caps_rad_s=velocity_caps.tolist(),
                      gripper_feedback=grip, right_tcp_mm=position.tolist(),
                      right_tcp_wxyz=quaternions[0, 1].tolist())
        if approach_only:
            targets = approach_target(frames, mask, report['depth'], position, quaternions[0, 1])
        else:
            points = measure(frames, mask, report['depth'])
            targets = waypoints(points, position, rotation, offset, parameters)
        report.update(targets, status='waypoints_ready')
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
        report.update(reason=str(error))
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report
