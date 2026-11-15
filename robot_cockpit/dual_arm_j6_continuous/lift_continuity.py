"""Process-local lift constraints; shared planners and physical limits stay unchanged."""
from contextlib import contextmanager
import json
import numpy as np

MAX_JOINT_EXCURSION_DEG = 50.0


def lift_start(p):
    if p.target_description not in ('book_plug_z', 'book_grasp_lift'):
        return None
    q = np.asarray(p.arm_joint_position, float).reshape(14)
    xyz, _ = p._v31_bridge.tcp(q)
    target = np.asarray(p.target_arm_tool_position[1], float)
    if (target[2] - xyz[0, 1, 2] <= .001
            or np.linalg.norm(target[:2] - xyz[0, 1, :2]) > .003):
        return None
    return q.copy()


def local_bounds(bounds, q):
    bounds, q = np.asarray(bounds, float), np.asarray(q, float)
    if (bounds.shape != (2, 14) or q.shape != (14,)
            or not np.isfinite(bounds).all() or not np.isfinite(q).all()
            or np.any(bounds[0] >= bounds[1]) or np.any(q < bounds[0]) or np.any(q > bounds[1])):
        raise ValueError('Lift continuity: invalid measured start or joint limits')
    result = bounds.copy()
    radius = np.deg2rad(MAX_JOINT_EXCURSION_DEG)
    result[0, 7:] = np.maximum(bounds[0, 7:], q[7:] - radius)
    result[1, 7:] = np.minimum(bounds[1, 7:], q[7:] + radius)
    return result


def path_metrics(samples, q):
    a = np.asarray(samples, float)
    if a.ndim != 2 or a.shape[1] != 14 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Lift continuity: invalid path samples')
    excursions = np.rad2deg(np.max(abs(a[:, 7:] - q[7:]), axis=0))
    left_drift = float(np.max(abs(a[:, :7] - q[:7])))
    return dict(accepted=bool(excursions.max() <= MAX_JOINT_EXCURSION_DEG + 1e-5
                              and left_drift <= .002),
                right_joint_excursion_deg=excursions.tolist(),
                max_joint='right J'+str(int(np.argmax(excursions))+1),
                limit_deg=MAX_JOINT_EXCURSION_DEG, left_drift_rad=left_drift)


def source_metrics(start, waypoints, raw, panel, move_arm):
    """Raw solver left-arm redundancy is not a command; inspect published hold.

    The base planner freezes the left arm in limited_waypoints and in
    _get_next_planned_arm_cmd. Do not rewrite source arrays to hide actual drift.
    """
    sources = {name: path_metrics(values, start) for name, values in
               [('waypoints', waypoints), ('motiongen_raw', raw), ('rt_interpolation', panel)]}
    published = path_metrics(np.concatenate([start[None], waypoints, panel]), start)
    reasons = []
    if list(move_arm) != [False, True]:
        reasons.append('lift_must_be_right_arm_only')
    if any(max(d['right_joint_excursion_deg']) > MAX_JOINT_EXCURSION_DEG + 1e-5
           for d in sources.values()):
        reasons.append('right_joint_neighbourhood_exceeded')
    if published['left_drift_rad'] > .002:
        reasons.append('published_left_joint_hold_violation')
    return dict(published, accepted=not reasons, rejection_reasons=reasons, sources=sources,
                raw_left_joint_drift_rad=sources['motiongen_raw']['left_drift_rad'],
                left_hold_basis='fixed_published_waypoints_and_rt_interpolation',
                raw_left_is_commanded=False)


def install(module):
    """Called only by independent J6 Planner; caller owns the GPU lock for bounds."""
    if getattr(module, '_j6_lift_continuity_installed', False):
        return
    original_bounds, original_validate = module.execution_bounds, module.validate_endpoint

    @contextmanager
    def bounds(p):
        q = lift_start(p)
        if q is None:
            with original_bounds(p):
                yield
            return
        old = p.arm_soft_joint_limits
        p._lift_continuity_diagnostics = dict(policy='measured_joint_neighbourhood',
            start_joints_rad=q.tolist(), limit_deg=MAX_JOINT_EXCURSION_DEG,
            accepted=False, phase='constrained_solve')
        try:
            p.arm_soft_joint_limits = local_bounds(old, q)
            # Existing execution_bounds updates IK/graph/trajopt tensors in place,
            # retains the 2-degree goal reserve and restores all tensors on exit.
            with original_bounds(p):
                yield
        finally:
            p.arm_soft_joint_limits = old

    def validate(p):
        if not original_validate(p):
            return False
        q = lift_start(p)
        if q is None:
            return True
        paths = []
        for arm in range(2):
            path, _ = p._v31_bridge.panel_path.sample(q[arm*7:arm*7+7],
                p.limited_waypoints[:, arm*7:arm*7+7], p.waypoints_dt,
                p._v31_bridge.limits[arm], sample_dt=.005)
            paths.append(path)
        n = max(map(len, paths))
        panel = np.concatenate([np.pad(a, ((0, n-len(a)), (0, 0)), mode='edge')
                                for a in paths], axis=1)
        raw = p._cmd_plan.position.detach().cpu().numpy()
        d = source_metrics(q, p.limited_waypoints, raw, panel, p._move_arm)
        d.update(phase='publication_path_check', samples=1+len(p.limited_waypoints)+len(raw)+len(panel),
                 policy='measured_joint_neighbourhood', start_joints_rad=q.tolist())
        p._lift_continuity_diagnostics = d
        print('Lift joint continuity: '+json.dumps(d), flush=True)
        return d['accepted']

    module.execution_bounds, module.validate_endpoint = bounds, validate
    module._j6_lift_continuity_installed = True
