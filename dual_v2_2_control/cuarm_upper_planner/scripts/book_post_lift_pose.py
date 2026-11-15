"""Scoped execution limits and handoff reserve for the one-command lift/place chain."""
from contextlib import contextmanager
import json
import numpy as np

# These actions are emitted only by the current book_grasp.sh motion policy.
ACTIONS = ('book_grasp_lift', 'book_post_lift_pose_restore', 'book_post_lift_pose_ccw',
           'book_grasp_shift', 'book_grasp_lower_hold', 'book_plug_x', 'book_plug_y', 'book_plug_z')
# Plan IK goals farther inside than the acceptance margin, so numerical
# trajectory refinement cannot silently consume all the handoff reserve.
SOLVER_RESERVE_DEG = 2.0
ENDPOINT_RESERVE_DEG = 1.0


def execution_limits(planner):
    limits = np.asarray(planner.arm_soft_joint_limits, float)
    if (limits.shape != (2, 14) or not np.isfinite(limits).all()
            or np.any(limits[0] >= limits[1])):
        raise ValueError('Invalid book execution joint limits')
    return limits


@contextmanager
def execution_bounds(planner):
    """Uniform execution constraints, with a separate interior IK goal reserve.

    Caller owns the GPU lock. Graph/trajectory START validity uses the actual
    execution bounds, including a legitimate measured start near a boundary.
    IK goal constraints and optimization costs seek interior configurations.
    Do not clip feedback or shrink the start-validity envelope at later stages.
    In-place edits preserve CUDA graph addresses; all changes are restored.
    """
    limits = execution_limits(planner)
    arm_names = list(planner.total_arm_joint_names)
    if len(arm_names) != 14:
        raise ValueError('Invalid book execution joint names')
    ik_rollouts = planner._motion_gen.ik_solver.get_all_rollout_instances()
    ik_ids = {id(r) for r in ik_rollouts}
    saved = {}
    for rollout in planner._motion_gen.get_all_rollout_instances():
        names = list(rollout.joint_names)
        for attr in ('bound_cost', 'bound_constraint'):
            cost = getattr(rollout, attr, None)
            tensor = getattr(getattr(cost, 'joint_limits', None), 'position', None)
            if tensor is None:
                continue
            if tuple(tensor.shape) != (2, len(names)):
                raise ValueError('Unexpected MotionGen joint-bound shape')
            interior = attr == 'bound_cost' or id(rollout) in ik_ids
            # Views sharing a storage region must receive the same policy.
            key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            if key in saved:
                if saved[key][3:] != (names, interior):
                    raise ValueError('MotionGen start and goal bound tensors unexpectedly alias')
                continue
            original = tensor.clone()
            desired = original.clone()
            margin = np.deg2rad(SOLVER_RESERVE_DEG) if interior else 0.
            for i in range(7, 14):
                index = names.index(arm_names[i])
                desired[0, index] = max(float(original[0, index]), limits[0, i]+margin)
                desired[1, index] = min(float(original[1, index]), limits[1, i]-margin)
                if desired[0, index] >= desired[1, index]:
                    raise ValueError('Empty book execution joint bounds')
            saved[key] = (tensor, original, desired, names, interior)
    if not saved or not ik_ids:
        raise ValueError('No MotionGen bounds for book planning')
    try:
        for tensor, _, desired, _, _ in saved.values():
            tensor.copy_(desired)
        yield
    finally:
        for tensor, original, _, _, _ in saved.values():
            tensor.copy_(original)


def joint_path_metrics(planner, points):
    """Check execution bounds on the actual RT curve, not just endpoints."""
    b = planner._v31_bridge
    start = np.asarray(planner.arm_joint_position, float).reshape(14)
    paths = []
    for arm in range(2):
        path, _ = b.panel_path.sample(start[arm*7:arm*7+7],
            points[:, arm*7:arm*7+7], planner.waypoints_dt, b.limits[arm], sample_dt=.005)
        paths.append(path)
    raw = planner._cmd_plan.position.detach().cpu().numpy()
    samples = np.concatenate((start[None, 7:], points[:, 7:], raw[:, 7:], paths[1]))
    limits = execution_limits(planner)[:, 7:]
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite book execution joint path')
    margins = np.minimum(samples-limits[0], limits[1]-samples)
    endpoint_margins = np.minimum(points[-1, 7:]-limits[0], limits[1]-points[-1, 7:])
    index = int(np.argmin(endpoint_margins))
    return dict(min_path_margin_deg=float(np.rad2deg(margins.min())),
                endpoint_joint_margin_deg=np.rad2deg(endpoint_margins).tolist(),
                min_endpoint_margin_deg=float(np.rad2deg(endpoint_margins[index])),
                limiting_joint='right J'+str(index+1),
                required_endpoint_margin_deg=ENDPOINT_RESERVE_DEG,
                solver_goal_reserve_deg=SOLVER_RESERVE_DEG,
                panel_samples=len(paths[1]))


def validate_endpoint(planner):
    """Reject a clipped or inaccurate solution before it can reach the panel."""
    raw = np.asarray(planner.waypoints, float)
    actual = np.asarray(planner.limited_waypoints, float)
    if (raw.shape != actual.shape or actual.ndim != 2 or actual.shape[1] != 14 or not len(actual)
            or not np.isfinite(raw).all() or not np.isfinite(actual).all()):
        raise ValueError('Invalid post-lift joint waypoints')
    clipped = float(np.max(np.abs(actual[:, 7:]-raw[:, 7:])))
    position, quat = planner._v31_bridge.tcp(actual[-1])
    target_p = np.asarray(planner.target_arm_tool_position[1], float)
    target_q = np.asarray(planner.target_arm_tool_orientation[1], float)
    q = np.asarray(quat[0, 1], float)
    pos_error = float(np.linalg.norm(position[0, 1]-target_p)*1000)
    norm = np.linalg.norm(q)*np.linalg.norm(target_q)
    angle = float(np.rad2deg(2*np.arccos(np.clip(abs(np.dot(q,target_q))/norm,0,1)))) if norm > 1e-8 else float('inf')
    margins = joint_path_metrics(planner, actual)
    accepted = (clipped <= 1e-6 and pos_error <= 1. and angle <= .25
                and margins['min_path_margin_deg'] >= 0.
                and margins['min_endpoint_margin_deg'] >= ENDPOINT_RESERVE_DEG)
    result = dict(action=planner.target_description, max_clipped_joint_delta_rad=clipped,
                  position_error_mm=pos_error, orientation_error_deg=angle, accepted=bool(accepted), **margins)
    planner._book_post_lift_diagnostics = result
    print('Book post-lift pose: '+json.dumps(result), flush=True)
    return accepted
