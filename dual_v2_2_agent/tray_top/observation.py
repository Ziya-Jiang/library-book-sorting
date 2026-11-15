"""Measured RGB-D evidence for an offline top-book separation step.

Depth samples enter in millimetres. Observations, rigid transforms and all
geometric limits use metres in one tray-aligned frame with +Z up. Callers must
associate visible landmarks on the actual book body, the layer beneath it and
the tray. A fitted cover plane alone provides no evidence of those surfaces.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np


class EvidenceError(ValueError):
    """The measured observations do not support a decision."""


@dataclass(frozen=True)
class TopSurface:
    center_m: np.ndarray
    normal: np.ndarray
    points_m: np.ndarray
    residual_m: float
    inlier_fraction: float


@dataclass(frozen=True)
class Observation:
    book_id: str
    head_token: str
    side_token: str
    head_time_s: float
    side_time_s: float
    top: TopSurface
    top_landmarks_m: Mapping[str, object]
    body_landmarks_m: Mapping[str, object]
    lower_landmarks_m: Mapping[str, object]
    tray_landmarks_m: Mapping[str, object]
    rgb: np.ndarray
    mask: np.ndarray
    rival_masks: tuple = ()


@dataclass(frozen=True)
class StepVerification:
    ok: bool
    reason: str
    progress_m: float | None = None
    cross_m: float | None = None
    yaw_deg: float | None = None
    body_residual_m: float | None = None
    lower_movement_m: float | None = None
    tray_movement_m: float | None = None


def _mask(value, shape):
    mask = np.asarray(value)
    if (mask.shape != shape or mask.ndim != 2
            or (mask.dtype != np.bool_ and not np.isin(mask, [0, 1, 255]).all())):
        raise EvidenceError("invalid_mask")
    return mask.astype(bool, copy=False)


def _scalar(value, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise EvidenceError("invalid_numeric_evidence")
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise EvidenceError("invalid_numeric_evidence")
    return float(value)


def _plane(points):
    center = np.mean(points, axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[-1]
    return center, normal if normal[2] >= 0 else -normal


def _spread(points, span):
    """Reject points confined to a line or to the depth-error scale."""
    centered = points - np.mean(points, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    return bool(np.ptp(centered @ axes[:2].T, axis=0).min() >= span)


def fit_top_surface(depth_mm, K, T_tray_camera_m, mask, *, inlier_m, min_points=30,
                    min_inlier_fraction=.85, min_span_m=.02, distortion=None,
                    min_valid_fraction=.4, max_tilt_deg=12.) -> TopSurface:
    """Back-project masked, registered depth and fit a supported top plane.

    K and distortion must describe the actual depth pixel grid. Pass no
    distortion for rectified depth. The returned points are observed inliers
    (bounded, deterministic subsampling), never an inferred underside or box.
    """
    inlier_m = _scalar(inlier_m)
    min_span_m = _scalar(min_span_m)
    max_tilt_deg = _scalar(max_tilt_deg, allow_zero=True)
    if (type(min_points) is not int or not 3 <= min_points <= 20000
            or not .5 < _scalar(min_inlier_fraction) <= 1
            or not _scalar(min_valid_fraction) <= 1 or max_tilt_deg >= 45):
        raise EvidenceError("invalid_plane_limits")
    depth = np.asarray(depth_mm, dtype=float)
    k = np.asarray(K, dtype=float)
    transform = np.asarray(T_tray_camera_m, dtype=float)
    if (depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) + abs(k[1, 0]) > 1e-8 or transform.shape != (4, 4)
            or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)
            or np.linalg.norm(transform[:3, 3]) > 10):
        raise EvidenceError("invalid_calibrated_depth_geometry")
    region = _mask(mask, depth.shape)
    y, x = np.nonzero(region & np.isfinite(depth) & (depth > 0) & (depth <= 10000))
    if len(x) < min_points or len(x) < min_valid_fraction * int(region.sum()):
        raise EvidenceError("insufficient_mask_depth")
    if len(x) > 20000:
        index = np.linspace(0, len(x) - 1, 20000, dtype=int)
        x, y = x[index], y[index]
    pixels = np.column_stack((x, y)).astype(float)
    d = None
    if distortion is not None:
        d = np.asarray(distortion, dtype=float).reshape(-1)
        if len(d) not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
            raise EvidenceError("invalid_depth_distortion")
    if d is None or not d.any():
        rays = np.linalg.solve(k, np.column_stack((pixels, np.ones(len(x)))).T).T
    else:
        try:
            import cv2
        except ImportError as error:
            raise EvidenceError("depth_distortion_backend_unavailable") from error

        try:
            rays_xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, d).reshape(-1, 2)
        except cv2.error as error:
            raise EvidenceError("invalid_depth_distortion") from error
        rays = np.column_stack((rays_xy, np.ones(len(x))))
    points = (rays * (depth[y, x] * .001)[:, None]) @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(points).all():
        raise EvidenceError("invalid_back_projection")

    # Select a planar consensus before least-squares fitting. Use fixed-seed,
    # bounded sampling so the same replay always produces the same result.
    sample = points[np.linspace(0, len(points) - 1, min(len(points), 1536), dtype=int)]
    rng = np.random.default_rng(0)
    best_count = 0
    best = None
    for _ in range(96):
        p, q, r = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(q - p, r - p)
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        normal /= length
        count = np.count_nonzero(np.abs((sample - p) @ normal) <= inlier_m)
        if count > best_count:
            best_count = count
            best = (p, normal)
    if best is None:
        raise EvidenceError("no_top_plane")
    center, normal = best
    for _ in range(2):
        inliers = np.abs((points - center) @ normal) <= inlier_m
        if int(inliers.sum()) < max(min_points, min_inlier_fraction * len(points)):
            raise EvidenceError("insufficient_top_plane_consensus")
        center, normal = _plane(points[inliers])
    residuals = np.abs((points - center) @ normal)
    inliers = residuals <= inlier_m
    measured = points[inliers]
    if (len(measured) < max(min_points, min_inlier_fraction * len(points))
            or not _spread(measured, min_span_m)):
        raise EvidenceError("insufficient_top_plane_extent")
    if normal[2] < np.cos(np.deg2rad(max_tilt_deg)):
        raise EvidenceError("top_plane_is_tilted")
    return TopSurface(center, normal, measured, float(residuals[inliers].max()), float(inliers.mean()))


def _visual_support(rgb, mask, other_rgb, other_mask):
    """Independent image evidence, following the batch mode's mutual ORB approach."""
    try:
        import cv2
    except ImportError as error:
        raise EvidenceError("appearance_backend_unavailable") from error

    try:
        return _orb_support(cv2, rgb, mask, other_rgb, other_mask)
    except cv2.error as error:
        raise EvidenceError("invalid_rgb_evidence") from error


def _orb_support(cv2, rgb, mask, other_rgb, other_mask):
    images = []
    masks = []
    for image, region in ((rgb, mask), (other_rgb, other_mask)):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise EvidenceError("invalid_rgb_evidence")
        images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        masks.append(_mask(region, image.shape[:2]))
    orb = cv2.ORB_create(nfeatures=2500, edgeThreshold=15, patchSize=15, fastThreshold=10)
    feature_sets = []
    for image, region in zip(images, masks):
        keypoints, descriptors = orb.detectAndCompute(image, region.astype(np.uint8) * 255)
        if descriptors is None:
            return False
        padded = np.pad(region.astype(np.uint8), 1)
        clearance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # ORB admits centres within its mask. Also exclude descriptors whose
        # rotated patch and smoothing footprint could see the background.
        interior = [i for i, keypoint in enumerate(keypoints)
                    if clearance[int(keypoint.pt[1]) + 1, int(keypoint.pt[0]) + 1] >= keypoint.size]
        if len(interior) < 8:
            return False
        feature_sets.append(([keypoints[i] for i in interior], descriptors[interior]))
    (ka, da), (kb, db) = feature_sets
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    good = [pair[0] for pair in matcher.knnMatch(da, db, k=2)
            if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    if len(good) < 8:
        return False
    a = np.float32([ka[m.queryIdx].pt for m in good])
    b = np.float32([kb[m.trainIdx].pt for m in good])
    homography, accepted = cv2.findHomography(a, b, cv2.RANSAC, 4.)
    if homography is None or accepted is None or not np.isfinite(homography).all():
        return False
    accepted = accepted.ravel().astype(bool)
    if int(accepted.sum()) < 8 or accepted.mean() < .65:
        return False
    for xy, mask in zip((a[accepted], b[accepted]), masks):
        y, x = np.nonzero(mask)
        mask_span = np.maximum([np.ptp(x), np.ptp(y)], 1)
        if (np.any(np.ptp(xy, axis=0) < .2 * mask_span)
                or cv2.contourArea(cv2.convexHull(xy)) < .03 * np.count_nonzero(mask)):
            return False
    return True


def _landmarks(record, name, span):
    if (not isinstance(record, Mapping) or len(record) < 3
            or any(not isinstance(k, str) or not k for k in record)):
        raise EvidenceError(f"{name}_reference_not_visible")
    points = np.asarray([record[k] for k in sorted(record)], dtype=float)
    if points.shape != (len(record), 3) or not np.isfinite(points).all() or not _spread(points, span):
        raise EvidenceError(f"{name}_reference_underdetermined")
    return points


def _validate_surfaces(observation, top_landmarks, body_landmarks, tolerance):
    top = observation.top
    center, normal, points = map(np.asarray, (top.center_m, top.normal, top.points_m))
    if (center.shape != (3,) or normal.shape != (3,) or points.ndim != 2 or points.shape[1] != 3
            or len(points) < 3 or not all(np.isfinite(v).all() for v in (center, normal, points))
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-5) or normal[2] <= 0
            or _scalar(top.residual_m, allow_zero=True) > tolerance
            or not .5 < _scalar(top.inlier_fraction) <= 1):
        raise EvidenceError("invalid_measured_top_surface")
    if (np.max(np.abs((points - center) @ normal)) > tolerance
            or np.max(np.abs((top_landmarks - center) @ normal)) > tolerance):
        raise EvidenceError("top_landmarks_disagree_with_depth")
    if any(np.linalg.norm(points - landmark, axis=1).min() > 2 * tolerance
           for landmark in top_landmarks):
        raise EvidenceError("top_landmarks_outside_measured_surface")
    body_heights = (body_landmarks - center) @ normal
    if (body_heights.min() >= -2 * tolerance or body_heights.max() > tolerance
            or body_heights.max() < -2 * tolerance or np.ptp(body_heights) <= 2 * tolerance):
        raise EvidenceError("body_not_distinct_below_cover")


def _snapshot(observation, rigid_m, stationary_m, max_view_skew_s):
    if not isinstance(observation.book_id, str) or not observation.book_id.strip():
        raise EvidenceError("missing_target_identity")
    tokens = (observation.head_token, observation.side_token)
    if any(not isinstance(t, str) or not t.strip() for t in tokens) or tokens[0] == tokens[1]:
        raise EvidenceError("missing_or_repeated_frame")
    times = [_scalar(observation.head_time_s), _scalar(observation.side_time_s)]
    if np.ptp(times) > max_view_skew_s:
        raise EvidenceError("stale_or_unsynchronised_observations")
    rgb = np.asarray(observation.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise EvidenceError("invalid_rgb_evidence")
    mask = _mask(observation.mask, rgb.shape[:2])
    if not mask.any():
        raise EvidenceError("top_book_mask_not_visible")
    for rival in observation.rival_masks:
        _mask(rival, rgb.shape[:2])
    groups = {}
    for name, span in (("tray", stationary_m), ("lower", stationary_m),
                       ("top", rigid_m), ("body", rigid_m)):
        groups[name] = _landmarks(getattr(observation, name + "_landmarks_m"),
                                 {"body": "solid_body", "lower": "lower_layer"}.get(name, name),
                                 2 * span)
    _validate_surfaces(observation, groups['top'], groups['body'], rigid_m)
    if np.max((groups['lower'] - observation.top.center_m) @ observation.top.normal) >= -2 * rigid_m:
        raise EvidenceError("lower_layer_not_distinct_below_top")
    return times, groups


def validate_observation(observation: Observation, *, rigid_m, stationary_m,
                         max_view_skew_s) -> None:
    """Fail closed before planning if the measured side/reference evidence is absent."""
    try:
        _snapshot(observation, _scalar(rigid_m), _scalar(stationary_m), _scalar(max_view_skew_s))
    except EvidenceError:
        raise
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        raise EvidenceError("invalid_observation") from error


def verify_step(before: Observation, after: Observation, direction, *, min_progress_m,
                max_progress_m, stationary_m, rigid_m, max_yaw_deg, max_cross_m,
                max_vertical_m, max_view_skew_s, max_interval_s,
                max_tilt_deg=3.) -> StepVerification:
    """Accept only a fresh, rigidly moved top book with still-visible fixed references."""
    metrics = {}

    def result(reason):
        return StepVerification(reason == "verified", reason, **metrics)

    try:
        for value in (min_progress_m, max_progress_m, stationary_m, rigid_m,
                      max_yaw_deg, max_cross_m, max_vertical_m, max_view_skew_s,
                      max_interval_s, max_tilt_deg):
            _scalar(value)
        if not stationary_m < min_progress_m <= max_progress_m or max_yaw_deg >= 45 or max_tilt_deg >= 45:
            return result("invalid_step_limits")
        axis = np.asarray(direction, dtype=float)
        if (axis.shape != (3,) or not np.isfinite(axis).all() or abs(axis[2]) > 1e-8
                or not np.isclose(np.linalg.norm(axis), 1., atol=1e-5)):
            return result("invalid_push_direction")
        if not isinstance(before.book_id, str) or not before.book_id.strip() or before.book_id != after.book_id:
            return result("target_identity_changed")
        tokens = (before.head_token, before.side_token, after.head_token, after.side_token)
        if any(not isinstance(t, str) or not t.strip() for t in tokens) or len(set(tokens)) != 4:
            return result("missing_or_repeated_frame")
        old_times, old_groups = _snapshot(before, rigid_m, stationary_m, max_view_skew_s)
        new_times, new_groups = _snapshot(after, rigid_m, stationary_m, max_view_skew_s)
        if min(new_times) <= max(old_times) or max(new_times) - min(old_times) > max_interval_s:
            return result("stale_or_unsynchronised_observations")
        for name in old_groups:
            if getattr(before, name + "_landmarks_m").keys() != getattr(after, name + "_landmarks_m").keys():
                return result(f"{name}_reference_not_visible")
        metrics['tray_movement_m'] = float(np.linalg.norm(new_groups['tray'] - old_groups['tray'], axis=1).max())
        if metrics['tray_movement_m'] > stationary_m:
            return result("tray_moved")
        metrics['lower_movement_m'] = float(np.linalg.norm(new_groups['lower'] - old_groups['lower'], axis=1).max())
        if metrics['lower_movement_m'] > stationary_m:
            return result("lower_layer_moved")
        # Determine motion from the solid side first. Fitting the cover and
        # body together can split a real cover shift between both residuals.
        a, b = old_groups['body'], new_groups['body']
        a_center, b_center = a.mean(axis=0), b.mean(axis=0)
        u, _, vh = np.linalg.svd((a - a_center).T @ (b - b_center))
        handedness = np.linalg.det(vh.T @ u.T)
        rotation = vh.T @ np.diag([1., 1., handedness]) @ u.T
        translation = b_center - rotation @ a_center
        errors = [np.linalg.norm(old_groups[surface] @ rotation.T + translation
                                 - new_groups[surface], axis=1).max()
                  for surface in ('body', 'top')]
        metrics['body_residual_m'] = float(max(errors))
        if metrics['body_residual_m'] > rigid_m:
            return result("cover_and_body_not_rigid")
        center = np.asarray(before.top.center_m)
        displacement = rotation @ center + translation - center
        progress = float(displacement @ axis)
        cross = float(np.linalg.norm(displacement[:2] - progress * axis[:2]))
        yaw = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
        tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1., 1.))))
        metrics.update(progress_m=progress, cross_m=cross, yaw_deg=yaw)
        if not min_progress_m <= progress <= max_progress_m:
            return result("unexpected_progress_or_stall")
        if cross > max_cross_m or abs(yaw) > max_yaw_deg:
            return result("off_axis_motion_or_yaw")
        if abs(displacement[2]) > max_vertical_m or tilt > max_tilt_deg:
            return result("book_lifted_or_tipped")
        expected_normal = rotation @ np.asarray(before.top.normal)
        angle = np.rad2deg(np.arccos(np.clip(expected_normal @ np.asarray(after.top.normal), -1., 1.)))
        predicted_top = np.asarray(before.top.points_m) @ rotation.T + translation
        after_plane_error = np.abs((predicted_top - after.top.center_m) @ after.top.normal)
        if angle > max_tilt_deg or np.percentile(after_plane_error, 95) > rigid_m:
            return result("top_plane_disagrees_with_solid_body")

        if not _visual_support(before.rgb, before.mask, after.rgb, after.mask):
            return result("same_book_appearance_unconfirmed")
        for observation, other in ((before, after), (after, before)):
            if any(_visual_support(observation.rgb, rival, other.rgb, other.mask)
                   for rival in observation.rival_masks):
                return result("ambiguous_book_appearance")
        return result("verified")
    except EvidenceError as error:
        return result(str(error))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return result("invalid_observation")
