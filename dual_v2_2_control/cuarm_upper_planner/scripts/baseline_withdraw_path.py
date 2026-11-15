"""Local Cartesian refinement of an already solved baseline withdrawal.

No robot I/O. Keep CuRobo's IK branch, add dense constrained waypoints, then
validate the complete deployed RT curve. This is not a fallback for collision,
invalid starts, failed IK or failed communication. All points are sent together.
"""
import json
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MAX_CORRECTION_RAD = np.deg2rad(10.)
SAMPLE_DT = .001


def eligible(p):
    d = getattr(p, '_book_lower_path_diagnostics', {})
    if (getattr(p, '_baseline_speed_factor', None) is None
            or p.target_description != 'book_plug_x'
            or d.get('accepted') or d.get('failure')):
        return False
    start, _ = p._v31_bridge.tcp(p.arm_joint_position)
    s = d.get('sources', {}).get('start', {})
    return (start[0, 1, 0] - p.target_arm_tool_position[1, 0] > .01
            and s.get('max_orientation_error_deg', float('inf')) <= .5
            and s.get('max_fixed_axis_error_mm', float('inf')) <= 3.
            and d.get('max_orientation_error_deg', float('inf')) <= 3.
            and d.get('max_fixed_axis_error_mm', float('inf')) <= 10.
            and d.get('advance_overshoot_mm', float('inf')) <= 3.)


def project_waypoints(tcp, start, points, goal, quaternion, bounds, checkpoint):
    """Small bounded joint corrections; no arbitrary IK branch change."""
    from book_post_lift_pose import SOLVER_RESERVE_DEG
    start, points, goal, bounds = [np.asarray(x, float) for x in (start, points, goal, bounds)]
    if (start.shape != (14,) or points.ndim != 2 or points.shape[1] != 14
            or len(points) < 2 or goal.shape != (3,) or bounds.shape != (2, 14)
            or not all(np.isfinite(x).all() for x in (start, points, goal, bounds))):
        raise ValueError('Invalid withdrawal projection inputs')
    initial, _ = tcp(start)
    distance = initial[0, 1, 0] - goal[0]
    if not .01 < distance <= .5:
        raise ValueError('Withdrawal refinement only supports 10–500mm along -X')
    count = max(24, int(np.ceil(distance / .01)))
    seeds = np.vstack([start, points])
    fractions = np.linspace(0., 1., count + 1)
    dense = np.array([np.interp(fractions, np.linspace(0., 1., len(seeds)), seeds[:, j])
                      for j in range(14)]).T
    target_rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])
    result = [None] * count
    max_correction = 0.
    endpoint_shift = np.zeros(7)
    # Solve the reserved endpoint first, then spread its redundant-joint
    # adjustment into the seeds across the whole path, rather than introducing
    # a large configuration change at the last short interval.
    for i in [count-1] + list(range(count-1)):
        checkpoint()
        original_seed = dense[i+1]
        seed = original_seed.copy()
        u = fractions[i+1]
        seed[7:] += (3*u*u - 2*u*u*u) * endpoint_shift
        position, _ = tcp(original_seed)
        xyz = goal.copy()
        # Preserve the solved path's progress profile; final endpoint is exact.
        xyz[0] = position[0, 1, 0] if i < count - 1 else goal[0]
        margin = np.deg2rad(SOLVER_RESERVE_DEG) if i == count - 1 else 0.
        rotation_weight = 3.5 if i == count - 1 else 1.
        lo = np.maximum(bounds[0, 7:] + margin, original_seed[7:] - MAX_CORRECTION_RAD)
        hi = np.minimum(bounds[1, 7:] - margin, original_seed[7:] + MAX_CORRECTION_RAD)
        if np.any(lo >= hi):
            raise ValueError('No bounded withdrawal correction with endpoint reserve')

        def residual(right):
            checkpoint()
            pos, quat = tcp(np.r_[start[:7], right])
            rot = Rotation.from_quat(quat[0, 1][[1, 2, 3, 0]])
            return np.r_[10. * (pos[0, 1] - xyz),
                         rotation_weight * (target_rotation * rot.inv()).as_rotvec(),
                         .0001 * (right - seed[7:])]

        # Clipping only initializes this bounded optimizer; its output still
        # requires independent FK, limits, collision and RT-path validation.
        solved = least_squares(residual, np.clip(seed[7:], lo, hi), bounds=(lo, hi),
                               max_nfev=40, ftol=1e-9, xtol=1e-9, gtol=1e-7)
        error = residual(solved.x)
        # The endpoint retains its existing 1mm acceptance envelope; aim for
        # 0.95mm to leave numerical reserve while keeping the 2deg joint reserve.
        position_limit = .0095 if i == count - 1 else .001
        angle_error = np.linalg.norm(error[3:6]) / rotation_weight
        angle_limit = .24 if i == count - 1 else .05
        if not solved.success or np.linalg.norm(error[:3]) > position_limit or angle_error > np.deg2rad(angle_limit):
            raise ValueError(f'Withdrawal waypoint {i+1}/{count} projection did not converge: '
                             f'status={solved.status}, position_mm={np.linalg.norm(error[:3])*100:g}, '
                             f'angle_deg={np.rad2deg(angle_error):g}')
        max_correction = max(max_correction, float(np.max(np.abs(solved.x - original_seed[7:]))))
        if i == count-1:
            endpoint_shift = solved.x - original_seed[7:]
        result[i] = np.r_[start[:7], solved.x]
    return np.array(result), max_correction


def dynamics_limits(p):
    limits = p._motion_gen.kinematics.get_joint_limits()
    order = [limits.joint_names.index(n) for n in p.total_arm_joint_names]
    values = []
    for name in ('velocity', 'acceleration', 'jerk'):
        a = getattr(limits, name).detach().cpu().numpy()
        values.append(np.minimum(abs(a[0, order]), abs(a[1, order])))
    # .limits is the 10%-scaled leading-joint timing request. Check every
    # joint against the actual backend limits as well as the model limits.
    values[0] = np.minimum(values[0], p._v31_bridge.joint_velocity_limits.reshape(14))
    values = np.array(values)
    if values.shape != (3, 14) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Missing withdrawal dynamics limits')
    return values


def sample_panel(p, points, dt):
    start = np.asarray(p.arm_joint_position)
    paths = [p._v31_bridge.panel_path.sample(start[a*7:a*7+7], points[:, a*7:a*7+7],
             dt, p._v31_bridge.limits[a], sample_dt=SAMPLE_DT)[0] for a in range(2)]
    count = max(map(len, paths))
    return np.concatenate([np.pad(x, ((0, count-len(x)), (0, 0)), mode='edge') for x in paths], axis=1)


def dynamics_ratios(panel, limits):
    # Include rest on both ends; catch start/finish discontinuities as well as
    # internal knot spikes. Conservative finite differences on the actual curve.
    derivative = np.pad(panel, ((3, 3), (0, 0)), mode='edge')
    ratios = []
    for limit in limits:
        derivative = np.diff(derivative, axis=0) / SAMPLE_DT
        ratios.append(float(np.max(np.abs(derivative) / limit)))
    return np.array(ratios)


def refine(p):
    """Transactional; rejected candidates never replace the old unsent path."""
    if not eligible(p):
        return False
    from curobo.types.state import JointState
    from book_lower_pose import validate_lower_path
    from book_post_lift_pose import validate_endpoint, execution_limits
    began = time.monotonic()
    fields = ('waypoints', 'limited_waypoints', 'waypoints_dt', '_cmd_plan',
              'waypoints_velocity', 'waypoints_acceleration', 'waypoints_jerk')
    saved = {name: getattr(p, name) for name in fields}
    original = p._book_lower_path_diagnostics
    accepted = False
    diagnostic = dict(event='baseline_withdraw_refinement', original=original, accepted=False)

    def checkpoint():
        if time.monotonic() - began > 3.:
            raise ValueError('Withdrawal refinement planning budget exhausted')
        p.raise_if_state_thread_failed()

    try:
        if np.max(np.abs(p.waypoints[:, 7:] - p.limited_waypoints[:, 7:])) > 1e-6:
            raise ValueError('Refinement cannot repair clipped joint paths')
        points, correction = project_waypoints(p._v31_bridge.tcp, p.arm_joint_position,
            p.limited_waypoints, p.target_arm_tool_position[1], p.target_arm_tool_orientation[1],
            execution_limits(p), checkpoint)
        # Retain original total nominal time; stretch only if the new actual
        # curve would exceed the stricter RT/model v/a/j limits.
        dt = float(p.waypoints_dt) * (len(p.limited_waypoints) + 1) / len(points)
        limits = dynamics_limits(p)
        diagnostic['retiming'] = []
        for _ in range(4):
            checkpoint()
            panel = sample_panel(p, points, dt)
            ratios = dynamics_ratios(panel, limits)
            diagnostic['retiming'].append(dict(dt=dt, ratios=ratios.tolist(), samples=len(panel)))
            if np.all(ratios <= 1.):
                break
            dt *= max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2])) * 1.1
        else:
            raise ValueError('Withdrawal retiming failed dynamics limits')
        checkpoint()
        p.waypoints = points.copy()
        p.limited_waypoints = points.copy()
        p.waypoints_dt = dt
        # Dense command samples now describe the refined RT curve, not the
        # rejected original MotionGen path. Keep its diagnostics separately.
        command = panel[np.r_[np.arange(0, len(panel)-1, 20), len(panel)-1]]
        p._cmd_plan = JointState.from_position(p.tensor_args.to_device(command),
                                               joint_names=p.total_arm_joint_names)
        p.waypoints_velocity = np.gradient(points, dt, axis=0)
        p.waypoints_acceleration = np.gradient(p.waypoints_velocity, dt, axis=0)
        p.waypoints_jerk = np.gradient(p.waypoints_acceleration, dt, axis=0)
        p._baseline_refined_path = True
        valid = bool(validate_lower_path(p, advance_axis=0) and validate_endpoint(p))
        checkpoint()
        accepted = valid
        diagnostic.update(accepted=accepted, waypoint_count=len(points), waypoint_dt=dt,
            max_joint_correction_deg=float(np.rad2deg(correction)), dynamics_ratios=ratios.tolist(),
            refined=p._book_lower_path_diagnostics)
    except ValueError as error:
        diagnostic['failure'] = str(error)
    finally:
        p._baseline_refined_path = False
        if not accepted:
            for name, value in saved.items():
                setattr(p, name, value)
        diagnostic['elapsed_seconds'] = time.monotonic() - began
        p._baseline_withdraw_diagnostics = diagnostic
        print('Baseline withdrawal: ' + json.dumps(diagnostic, allow_nan=False), flush=True)
    return accepted
