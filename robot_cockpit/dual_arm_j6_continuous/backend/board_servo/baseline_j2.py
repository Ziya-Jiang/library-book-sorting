"""Opt-in baseline right-J2 preference in the exact task Jacobian null space.

Preserve the primary bounded solution's instantaneous TCP velocity. A single
coordinated null-space increment redistributes joint motion, subject to EVERY
original velocity/acceleration/position/lead/braking bound. No component-wise
clipping, limit changes, integrator reset, or independent joint commands.
"""
import numpy as np
from .joint_limits import JointLimitedStep, bounded_velocity, velocity_interval

POLICY = 'baseline_right_j2_v1'
J2 = 1  # Index in the right-arm seven-joint solve, never the fourteen-joint state.
BAND_RAD = np.deg2rad(10.)
RETURN_RAD_S = .1


def redistribute(a, velocity, command, measured, previous, bounds, c, dt):
    lo, hi = bounds
    band = min(BAND_RAD, (hi[J2]-lo[J2])/4.)
    low = min(command[J2], measured[J2])-lo[J2]
    high = hi[J2]-max(command[J2], measured[J2])
    margin = min(low, high)
    info = dict(policy=POLICY, margin_deg=float(np.rad2deg(margin)),
                primary_j2_rad_s=float(velocity[J2]),
                selected_j2_rad_s=float(velocity[J2]), applied=False)
    if margin >= band:
        return velocity, info
    # Smoothly activate inside the existing ten-degree preference band. Keep
    # an already stronger inward primary velocity; never resist a safe retreat.
    activation = float(np.clip(1.-margin/band, 0., 1.))
    strength = activation*activation*(3.-2.*activation)
    inward = (1. if low < high else -1.)*RETURN_RAD_S*strength
    direction = 1. if low < high else -1.
    if direction*velocity[J2] >= direction*inward:
        return velocity, info
    desired = velocity[J2]+strength*(inward-velocity[J2])
    # SVD gives an exact linear task null space, unlike a damped pseudo-inverse
    # projector which can trade tool pose for the secondary posture objective.
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    tolerance = max(a.shape)*np.finfo(float).eps*max(float(singular[0]), 1.)
    rank = int(np.count_nonzero(singular > tolerance))
    basis = vh[rank:].T
    if not basis.size:
        info['blocked'] = 'no_task_redundancy'
        return velocity, info
    vector = basis @ basis[J2]
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        info['blocked'] = 'j2_not_in_task_null_space'
        return velocity, info
    vector /= length
    lower, upper, *_ = velocity_interval(command, measured, previous, bounds, c, dt)
    zlo, zhi = -np.inf, np.inf
    for i, component in enumerate(vector):
        if abs(component) < 1e-14:
            continue
        edges = ((lower[i]-velocity[i])/component, (upper[i]-velocity[i])/component)
        zlo, zhi = max(zlo, min(edges)), min(zhi, max(edges))
    # Zero is the already accepted primary solution. If roundoff puts it outside
    # the secondary interval, retain it; never relax a bound to create redundancy.
    if zlo > 0. or zhi < 0. or zlo > zhi:
        info['blocked'] = 'original_bounds'
        return velocity, info
    shift = float(np.clip((desired-velocity[J2])/vector[J2], zlo, zhi))
    shift *= 1.-1e-9  # Stay inside the scalar interval without clipping joints.
    selected = velocity+vector*shift
    task_change = float(np.max(np.abs(a @ (selected-velocity))))
    if (not np.isfinite(selected).all() or task_change > 1e-10
            or np.any(selected < lower-1e-12) or np.any(selected > upper+1e-12)):
        raise ValueError('Baseline J2 redistribution violated primary task/bounds')
    info.update(selected_j2_rad_s=float(selected[J2]), applied=bool(abs(shift)>1e-12),
                task_velocity_change=task_change)
    return selected, info


class BaselineJ2Step(JointLimitedStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.j2_diagnostics = None

    def reset(self):
        super().reset()
        self.j2_diagnostics = None

    def solve_velocity(self, a, target, command, measured, previous, bounds, c, dt):
        primary = bounded_velocity(a, target, command, measured, previous, bounds, c, dt)
        result, self.j2_diagnostics = redistribute(
            a, primary, command, measured, previous, bounds, c, dt)
        return result
