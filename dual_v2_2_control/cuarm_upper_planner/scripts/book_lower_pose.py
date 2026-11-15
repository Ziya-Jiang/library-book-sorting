"""Opt-in constant-orientation descent for book_grasp.sh; no robot I/O.

CuRobo's main EE is the left arm. This installed version does not implement
include_link_pose, so constrain the right link's actual PoseCost explicitly.
The metric is an optimization cost; the independently reconstructed RT path
must also pass before any waypoints are published.
"""
from contextlib import contextmanager
import json
import numpy as np

ACTION = 'book_grasp_lower_hold'
PLANNED_ANGLE_DEG = 0.5
BASELINE_ANGLE_DEG = 0.5
PLANNED_XY_MM = 3.0


@contextmanager
def right_link_hold(planner, advance_axis=2, *, continuous_baseline=False):
    """Caller owns _gpu_lock; restore cost tensors even on a solver exception."""
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    if advance_axis not in (0, 1, 2):
        raise ValueError('Invalid constrained Cartesian advance axis')
    right = planner.target_link_names[1]
    costs = []
    for rollout in planner._motion_gen.get_all_pose_rollout_instances():
        if getattr(getattr(rollout, 'kinematics', None), 'ee_link', None) == right:
            cost = getattr(rollout, 'goal_cost', None)
        else:
            cost = getattr(rollout, '_link_pose_costs', {}).get(right)
        if cost is not None and all(cost is not x for x in costs):
            costs.append(cost)
    if not costs or not any(getattr(c, 'enabled', False) for c in costs):
        raise ValueError('Lowering requires an enabled right-link CuRobo pose cost')
    # Save all first: some rollout cost tensors share configuration storage.
    saved = [(c, c.run_vec_weight.clone(), c.project_distance_tensor.clone()) for c in costs]
    # Sparse long baseline moves need more than the old short-step pose cost.
    # This changes the optimization objective, NOT the independent acceptance
    # limits. Ordinary/plug callers retain their existing objective.
    weight = 20. if continuous_baseline else 1.
    # Only an explicitly opted-in independent planner changes plug Z costs.
    if advance_axis == 2 and getattr(planner, "target_description", "") == "book_plug_z":
        weight = float(getattr(planner, "_trial_z_hold_weight", weight))
        if not np.isfinite(weight) or weight <= 0.:
            raise ValueError("Invalid independent Z hold weight")
    weights = [weight]*6
    weights[3+advance_axis] = 0.
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=planner.tensor_args.to_device(weights),
        project_to_goal_frame=False)
    try:
        for c in costs:
            c.update_metric(metric, update_offset_waypoint=False)
        print(f'Book Cartesian hold: right link {right}; free base {"XYZ"[advance_axis]}, hold other axes and orientation; hold_weight={weight:g}', flush=True)
        yield
    finally:
        for c, weights, projection in saved:
            c.run_vec_weight.copy_(weights)
            c.project_distance_tensor.copy_(projection)


def path_metrics(position, quaternion, goal_position, goal_quaternion, advance_axis=2):
    """Scalar-first quaternions; q and -q represent the same orientation."""
    p = np.asarray(position, float)
    q = np.asarray(quaternion, float)
    gp, gq = np.asarray(goal_position, float), np.asarray(goal_quaternion, float)
    if (p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4)
            or not len(p) or gp.shape != (3,) or gq.shape != (4,)
            or not all(np.isfinite(x).all() for x in (p, q, gp, gq))):
        raise ValueError('Invalid lowering FK samples')
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-8) or np.linalg.norm(gq) < 1e-8:
        raise ValueError('Invalid lowering quaternion')
    dots = np.abs((q/norms[:, None]) @ (gq/np.linalg.norm(gq)))
    angles = np.rad2deg(2*np.arccos(np.clip(dots, 0., 1.)))
    fixed = [i for i in range(3) if i != advance_axis]
    error = float(np.max(np.linalg.norm(p[:, fixed]-gp[fixed], axis=1))*1000)
    return dict(samples=len(p), max_orientation_error_deg=float(angles.max()),
                max_xy_error_mm=error, max_fixed_axis_error_mm=error,
                fixed_axes=''.join('XYZ'[i] for i in fixed))


def validate_lower_path(planner, advance_axis=2):
    """Check clipped waypoints and the deployed RT interpolation at 5 ms.

    Existing planning-start freshness, handoff, joint-limit and collision
    protections still apply. This is model validation, not measured tracking.
    """
    b = planner._v31_bridge
    angle_limit = (BASELINE_ANGLE_DEG if getattr(planner, '_baseline_speed_factor', None) is not None
                   else PLANNED_ANGLE_DEG)
    points = np.asarray(planner.limited_waypoints, float)
    start = np.asarray(planner.arm_joint_position, float).reshape(-1)
    raw = planner._cmd_plan.position.detach().cpu().numpy()
    if points.ndim != 2 or points.shape[1] != 14 or start.shape != (14,):
        raise ValueError('Invalid lowering joint path')
    # The left arm remains fixed, exactly as it will be sent to the panel.
    paths = []
    for arm in range(2):
        path, _ = b.panel_path.sample(start[arm*7:arm*7+7],
            points[:, arm*7:arm*7+7], planner.waypoints_dt, b.limits[arm], sample_dt=.005)
        paths.append(path)
    n = max(map(len, paths))
    panel = np.concatenate([np.pad(p, ((0, n-len(p)), (0, 0)), mode='edge') for p in paths], axis=1)
    samples = np.concatenate((start[None, :], points, raw, panel))
    position, quat = b.tcp(samples)
    result = path_metrics(position[:, 1], quat[:, 1],
                          planner.target_arm_tool_position[1], planner.target_arm_tool_orientation[1], advance_axis)
    # Keep the source of deviations: a combined maximum cannot tell whether
    # the solver, waypoint clipping or the deployed interpolation is at fault.
    sources = {}
    offset = 0
    command_source = 'refined_command' if getattr(planner, '_baseline_refined_path', False) else 'motiongen'
    for name, size in (('start', 1), ('waypoints', len(points)),
                       (command_source, len(raw)), ('rt_interpolation', len(panel))):
        end = offset + size
        sources[name] = path_metrics(position[offset:end, 1], quat[offset:end, 1],
            planner.target_arm_tool_position[1], planner.target_arm_tool_orientation[1], advance_axis)
        offset = end
    result['sources'] = sources
    is_plug = getattr(planner, 'target_description', '') in ('book_plug_x', 'book_plug_y', 'book_plug_z') or advance_axis != 2
    if is_plug:
        # No large overshoot beyond the start/goal, including vertical placement.
        lo, hi = sorted((position[0, 1, advance_axis], planner.target_arm_tool_position[1][advance_axis]))
        observed = position[:, 1, advance_axis]
        result['advance_overshoot_mm'] = max(0., lo-observed.min(), observed.max()-hi)*1000.
    result.update(event='lowering_path_check', right_link=planner.target_link_names[1],
                  orientation_limit_deg=angle_limit, xy_limit_mm=PLANNED_XY_MM,
                  panel_samples=len(panel), accepted=False)
    if (result.get('advance_overshoot_mm', 0.) <= 3.
            and result['max_orientation_error_deg'] <= angle_limit
            and result['max_xy_error_mm'] <= PLANNED_XY_MM):
        # The panel curve differs from MotionGen interpolation. Check its
        # joint limits and collision constraints as well as its orientation.
        grip = list(planner.target_gripper_joint)
        order = np.asarray(planner.reorder_input(np.arange(14),
                            list(range(14, 14+len(grip))), as_curobo=True), int)
        values = np.concatenate((panel, np.broadcast_to(grip, (len(panel), len(grip)))), axis=1)[:, order]
        with planner._gpu_lock:
            result['accepted'] = True
            for i in range(0, len(values), 1024):
                metrics = planner._motion_gen.rollout_fn.rollout_constraint(
                    planner.tensor_args.to_device(np.ascontiguousarray(values[i:i+1024])).unsqueeze(1),
                    use_batch_env=False)
                if not bool(metrics.feasible.all().item()):
                    result.update(accepted=False, failure='panel_collision_or_joint_limit')
                    break
    if is_plug:
        result.update(event='plug_axis_path_check', advance_axis='XYZ'[advance_axis])
    print('Book Cartesian path: '+json.dumps(result, allow_nan=False), flush=True)
    planner._book_lower_path_diagnostics = result
    return result['accepted']
