"""Bounded summaries of failed ordinary book plans; never alter feasibility."""
import numpy as np
from v4_1_endpoint_diagnostics import diagnose_endpoint


def host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def summary(value):
    if value is None:
        return None
    a = host(value)
    if a.dtype.kind == 'b':
        return dict(shape=list(a.shape), passed=int(a.sum()), failed=int((~a).sum()))
    finite = a[np.isfinite(a)]
    return dict(shape=list(a.shape), nonfinite=int(a.size-finite.size),
                minimum=float(finite.min()) if finite.size else None,
                maximum=float(finite.max()) if finite.size else None)


def constraints(motion):
    """Snapshot live tensors while execution_bounds/right_link_hold are active."""
    out = []
    try:
        for index, rollout in enumerate(motion.get_all_rollout_instances()):
            item = dict(rollout=index, names=list(rollout.joint_names))
            for attr in ('bound_cost', 'bound_constraint'):
                cost = getattr(rollout, attr, None)
                bounds = getattr(cost, 'joint_limits', None)
                data = {}
                for field in ('position', 'velocity', 'acceleration', 'jerk'):
                    value = getattr(bounds, field, None)
                    if value is not None:
                        data[field] = host(value).tolist()
                for field in ('weight', 'activation_distance'):
                    value = getattr(cost, field, None)
                    if value is not None:
                        data[field] = host(value).tolist()
                item[attr] = data
            item['link_pose_costs'] = {}
            for name, cost in getattr(rollout, '_link_pose_costs', {}).items():
                item['link_pose_costs'][name] = {key: host(getattr(cost, key)).tolist()
                    for key in ('weight', 'vec_weight', 'run_vec_weight', 'project_distance_tensor')
                    if getattr(cost, key, None) is not None}
            for attr in ('pose_cost', 'self_collision_constraint', 'primitive_collision_constraint'):
                cost = getattr(rollout, attr, None)
                if cost is not None:
                    item[attr] = {key: host(getattr(cost, key)).tolist()
                                  for key in ('weight', 'threshold', 'activation_distance')
                                  if getattr(cost, key, None) is not None}
            out.append(item)
        return out
    except Exception as error:
        return dict(unavailable=f'{type(error).__name__}: {error}', partial=out)


def result_summary(result):
    out = dict(status=str(result.status), success=bool(result.success.item()),
               attempts=int(getattr(result, 'attempts', 0)))
    for key in ('ik_time', 'graph_time', 'trajopt_time', 'finetune_time', 'solve_time', 'total_time'):
        value = getattr(result, key, None)
        out[key] = float(value) if value is not None and np.isfinite(float(value)) else None
    for key in ('position_error', 'rotation_error', 'cspace_error', 'optimized_dt'):
        out[key] = summary(getattr(result, key, None))
    return out


def bound_violations(values, lower, upper, names, maximum=12):
    values = host(values).reshape(-1, len(names))
    lo, hi = np.asarray(lower), np.asarray(upper)
    bad = np.maximum(lo-values, values-hi)
    ids = np.argwhere((bad > 0) | ~np.isfinite(values))
    return dict(total=int(len(ids)), examples=[dict(sample=int(row), joint=names[col],
        value=float(values[row, col]) if np.isfinite(values[row, col]) else None,
        lower=float(lo[col]), upper=float(hi[col]),
        excess=float(bad[row, col]) if np.isfinite(bad[row, col]) else None)
        for row, col in ids[:maximum]])


def selection_evidence(traj, solver):
    """Reconstruct installed CuRobo jit_feasible_success/best_select (read only).

    A finite 10000 failure penalty can lose to a feasible candidate's large
    smoothness cost. Report both winners rather than claiming the goal is IK-infeasible.
    """
    m = traj.metrics
    feasible = host(m.feasible).all(axis=-1)
    if m.position_error is not None and m.rotation_error is not None:
        converged = ((host(m.position_error)[..., -1] <= solver.position_threshold) &
                     (host(m.rotation_error)[..., -1] <= solver.rotation_threshold))
    elif m.cspace_error is not None:
        converged = host(m.cspace_error)[..., -1] <= solver.cspace_threshold
    else:
        return dict(available=False, reason='no convergence metric')
    smooth = host(traj.smooth_label)
    valid = feasible & converged & smooth
    pose = getattr(m, 'pose_error', None)
    error = host(pose if pose is not None else m.cspace_error)[..., -1]
    running = host(m.cost).mean(axis=-1)*.0001
    smooth_cost = host(traj.smooth_error)
    base = error+smooth_cost+running
    ranked = base+(~valid)*10000.
    if ranked.ndim != 1 or len(ranked) > 32:
        return dict(available=False, reason='not a single query of <=32 seeds')
    chosen = int(ranked.argmin())
    valid_ids = np.flatnonzero(valid)
    best_valid = int(valid_ids[np.argmin(base[valid_ids])]) if len(valid_ids) else None
    return dict(available=True, source='installed CuRobo finite penalty ranking, independently reconstructed',
        feasible=feasible.tolist(), converged=converged.tolist(), smooth=smooth.tolist(),
        valid=valid.tolist(), smooth_cost=smooth_cost.tolist(), running_cost=running.tolist(),
        convergence_cost=error.tolist(), ranked_cost=ranked.tolist(),
        selected_seed=chosen, best_valid_seed=best_valid,
        invalid_selected_despite_valid=bool(not valid[chosen] and len(valid_ids)))


def failure_evidence(planner, collision, names, start, goal, result):
    out = dict(unproven='Failure status alone does not identify a collision or an unreachable goal.')
    try:
        out['start'] = diagnose_endpoint(collision, names, host(start))
        if goal is not None:
            out['joint_goal'] = diagnose_endpoint(collision, names, host(goal))
        out['execution_joint_names'] = list(planner.total_arm_joint_names)
        out['execution_joint_bounds_rad'] = np.asarray(planner.arm_soft_joint_limits).tolist()
        out['solver_thresholds'] = {}
        for name in ('ik_solver', 'trajopt_solver', 'finetune_trajopt_solver'):
            solver = getattr(planner._motion_gen, name, None)
            out['solver_thresholds'][name] = {field: float(getattr(solver, field))
                for field in ('position_threshold', 'rotation_threshold', 'cspace_threshold')
                if getattr(solver, field, None) is not None}
        out['optimizer_stages'] = {}
        debug = result.debug_info or {}
        for stage in ('trajopt_result', 'finetune_trajopt_result'):
            traj = debug.get(stage)
            if traj is None:
                out['optimizer_stages'][stage] = dict(available=False)
                continue
            metrics = getattr(traj, 'metrics', None)
            data = dict(available=True, success=summary(getattr(traj, 'success', None)),
                        smooth=summary(getattr(traj, 'smooth_label', None)),
                        optimized_dt_s=summary(getattr(traj, 'optimized_dt', None)),
                        dt_exception=str((getattr(traj, 'debug_info', None) or {}).get('dt_exception')))
            for field in ('position_error', 'rotation_error', 'cspace_error', 'smooth_error'):
                value = getattr(traj, field, None)
                data['selected_'+field] = summary(value)
                if value is not None and host(value).size <= 32:
                    data['selected_'+field+'_values'] = [float(v) if np.isfinite(v) else None
                                                        for v in host(value).reshape(-1)]
            # Padded rollout arrays are not terminal errors. Keep selected
            # scalar errors above separate from whole-rollout summaries below.
            for field in ('feasible', 'position_error', 'rotation_error', 'cspace_error', 'constraint'):
                data[field] = summary(getattr(metrics, field, None))
            solver = getattr(planner._motion_gen,
                'finetune_trajopt_solver' if stage.startswith('finetune') else 'trajopt_solver', None)
            if getattr(traj, 'smooth_label', None) is not None and getattr(traj, 'smooth_error', None) is not None:
                try:
                    data['selection'] = selection_evidence(traj, solver)
                except Exception as error:
                    data['selection'] = dict(available=False, reason=str(error))
            solution = getattr(traj, 'solution', None)
            if solution is not None and getattr(solution, 'position', None) is not None:
                ordered = solution.get_ordered_joint_state(names)
                points = host(ordered.position).reshape(-1, len(names))
                data['model_position_bounds'] = bound_violations(points, *collision.bounds, names)
                arm_indices = [names.index(n) for n in planner.total_arm_joint_names]
                data['execution_position_bounds'] = bound_violations(points[:, arm_indices],
                    *planner.arm_soft_joint_limits, planner.total_arm_joint_names)
                # Inspect at most 32 samples. Explicitly label sampling; not a full path proof.
                sample_ids = np.unique(np.linspace(0, len(points)-1, min(32, len(points)), dtype=int))
                bad = []
                for i in sample_ids:
                    endpoint = diagnose_endpoint(collision, names, points[i], max_contacts=3)
                    if not endpoint['cpu_geometry_feasible']:
                        bad.append(dict(sample=int(i), **endpoint))
                data['sampled_geometry'] = dict(total_points=len(points), checked=len(sample_ids),
                    infeasible_count=len(bad), examples=bad[:4], exhaustive=len(sample_ids)==len(points))
                # All returned seeds' final right TCP errors, separate from solver's quaternion metric.
                shape = host(ordered.position).shape
                finals = host(ordered.position).reshape(-1, shape[-2], len(names))[:, -1]
                errors = []
                for q in finals[:32]:
                    pos, quat = planner._v31_bridge.tcp(q[arm_indices])
                    dot = abs(float(np.dot(quat[0, 1], planner.target_arm_tool_orientation[1])))
                    errors.append(dict(position_mm=float(np.linalg.norm(pos[0, 1]-planner.target_arm_tool_position[1])*1000),
                        orientation_deg=float(np.degrees(2*np.arccos(np.clip(dot, 0, 1))))))
                data['right_tcp_terminal_errors'] = errors
                for field in ('velocity', 'acceleration', 'jerk'):
                    data[field+'_si'] = summary(getattr(ordered, field, None))
                    # Compare each sample against the actual optimizer limits.
                    value = getattr(ordered, field, None)
                    if value is not None:
                        solver = getattr(planner._motion_gen,
                            'finetune_trajopt_solver' if stage.startswith('finetune') else 'trajopt_solver', None)
                        rollout = getattr(solver, 'rollout_fn', None)
                        limits = getattr(getattr(getattr(rollout, 'bound_constraint', None), 'joint_limits', None), field, None)
                        if limits is not None and list(rollout.joint_names) == names:
                            data[field+'_bounds'] = bound_violations(value, *host(limits), names)
            out['optimizer_stages'][stage] = data
    except Exception as error:
        out['diagnostic_error'] = f'{type(error).__name__}: {error}'
    return out
