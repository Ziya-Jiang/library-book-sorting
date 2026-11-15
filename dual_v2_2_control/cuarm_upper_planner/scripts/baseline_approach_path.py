"""Baseline-only Cartesian path search. No robot I/O or feedback clipping.

CuRobo supplies collision-aware joint paths; bounded Cartesian projection
densifies them. Acceptance always checks the actual panel interpolation.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


class Rejected(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.details=details


@dataclass(frozen=True)
class Policy:
    budget_s: float = 8.
    seeds: int = 32
    spacing_m: float = .01
    goal_reserve_deg: float = 2.
    path_reserve_deg: float = 1.
    position_m: float = .003
    angle_deg: float = .5
    sample_dt: float = .001


def angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)) /
        (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)), 0., 1.)))


def seeds(start, bounds, count=32, reserve_deg=2.):
    """Deterministic local and broad seeds; never change the measured left arm."""
    lo, hi = bounds[:, 7:] + np.deg2rad(reserve_deg)*np.array([[1.], [-1.]])
    if np.any(lo >= hi):
        raise Rejected('Empty interior joint bounds')
    yield np.r_[start[:7], np.clip(start[7:], lo, hi)]
    samples = qmc.Sobol(7, scramble=False).random_base2(5)
    for i, sample in enumerate(samples[:count-1]):
        right = (np.clip(start[7:] + (sample*2-1)*np.deg2rad(25), lo, hi)
                 if i < 16 else lo+(hi-lo)*sample)
        yield np.r_[start[:7], right]


class Search:
    """Adapter provides FK, MotionGen seeds, RT sampling and collision tests."""
    def __init__(self, adapter, policy=Policy(), clock=time.monotonic):
        self.a, self.p, self.clock = adapter, policy, clock
        self.deadline = clock()+policy.budget_s
        self.rejections = []

    def check(self):
        self.a.check()
        if self.clock() >= self.deadline:
            raise TimeoutError('Baseline preflight exceeded 8s; no trajectory accepted')

    def project(self, seed, xyz, quat, *, reserve=2., reference=None):
        self.check()
        lo, hi = self.a.bounds[:, 7:] + np.deg2rad(reserve)*np.array([[1.], [-1.]])
        target = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        reference = seed[7:] if reference is None else reference
        def residual(q):
            self.check()
            pos, rot = self.a.tcp(np.r_[seed[:7], q])
            actual = Rotation.from_quat(rot[0, 1, [1, 2, 3, 0]])
            return np.r_[10*(pos[0, 1]-xyz), (target*actual.inv()).as_rotvec(),
                         .0001*(q-reference)]
        out = least_squares(residual, np.clip(seed[7:], lo, hi), bounds=(lo, hi),
                            max_nfev=45, ftol=1e-9, xtol=1e-9, gtol=1e-7)
        q = np.r_[seed[:7], out.x]
        pos, rot = self.a.tcp(q)
        if (not out.success or np.linalg.norm(pos[0, 1]-xyz) > .0005
                or angle(rot[0, 1], quat) > .1):
            raise Rejected('Cartesian projection did not converge',position_error_mm=float(np.linalg.norm(pos[0,1]-xyz)*1000),
                           orientation_error_deg=float(angle(rot[0,1],quat)),optimizer_status=int(out.status))
        return q

    def metrics(self, start, samples, xyz, quat, *, reshape=False):
        self.check()
        samples = np.asarray(samples, float)
        if samples.ndim != 2 or samples.shape[1] != 14 or not np.isfinite(samples).all():
            raise Rejected('Invalid RT samples')
        if not np.allclose(samples[:, :7], start[:7], atol=1e-8, rtol=0):
            raise Rejected('Nonmoving left arm changed')
        margins = np.minimum(samples-self.a.bounds[0], self.a.bounds[1]-samples)
        evidence=dict(joint_margin_deg=np.rad2deg(margins[:,7:].min(axis=0)).tolist(),
                      start_joint_rad=start.tolist(),end_joint_rad=samples[-1].tolist())
        if np.any(margins < 0):
            raise Rejected('RT curve exceeds execution joint bounds',**evidence)
        required = np.full(7, np.deg2rad(self.p.path_reserve_deg))
        if reshape:
            # A legal boundary start may only recover inward, never consume
            # more margin. Endpoint must meet the ordinary interior reserve.
            initial = np.minimum(start[7:]-self.a.bounds[0, 7:], self.a.bounds[1, 7:]-start[7:])
            required = np.minimum(required, initial)
        if np.any(margins[:, 7:] < required-1e-9):
            raise Rejected('Insufficient full-path joint reserve',**evidence)
        if np.min(margins[-1, 7:]) < np.deg2rad(self.p.path_reserve_deg):
            raise Rejected('Insufficient endpoint joint reserve',**evidence)
        pos, rot = self.a.tcp(samples)
        start_pos, _ = self.a.tcp(start)
        axes = [0, 1, 2] if reshape else [1, 2]
        fixed_error = float(np.linalg.norm(pos[:, 1, axes]-xyz[axes], axis=1).max())
        angular_error = float(angle(rot[:, 1], quat).max())
        evidence.update(max_fixed_axis_error_mm=fixed_error*1000,max_orientation_error_deg=angular_error)
        if fixed_error > self.p.position_m or angular_error > self.p.angle_deg:
            raise Rejected('RT curve violates frozen Cartesian pose',**evidence)
        if not reshape:
            x = pos[:, 1, 0]
            if (x.min() < start_pos[0, 1, 0]-.003 or x.max() > xyz[0]+.003
                    or np.max(np.maximum.accumulate(x)-x) > .003):
                raise Rejected('Advance curve reverses or overshoots')
        if np.linalg.norm(pos[-1, 1]-xyz) > .001 or angle(rot[-1, 1], quat) > .25:
            raise Rejected('Endpoint accuracy failed')
        try:self.a.collision(samples, self.check)
        except Rejected as error:
            raise Rejected(str(error),**evidence) from error
        return dict(min_path_margin_deg=float(np.rad2deg(margins[:, 7:].min())),
                    joint_margin_deg=np.rad2deg(margins[:, 7:].min(axis=0)).tolist(),
                    limiting_joint='right J'+str(int(np.argmin(margins[:, 7:].min(axis=0)))+1),
                    max_fixed_axis_error_mm=fixed_error*1000,
                    max_orientation_error_deg=angular_error, samples=len(samples))

    def path(self, start, goal_joint, xyz, quat, *, reshape=False):
        self.check()
        solved, nominal_dt = self.a.motion(start, goal_joint, self.deadline)
        solved = np.vstack([start, solved])
        start_pos, _ = self.a.tcp(start)
        n = max(2, int(np.ceil(abs(xyz[0]-start_pos[0, 1, 0])/self.p.spacing_m)),
                int(np.ceil(np.max(np.abs(goal_joint-start))/np.deg2rad(2))))
        if n > 250:
            raise Rejected('Too many constrained waypoints')
        points = []
        for u in np.linspace(0, 1, n+1)[1:]:
            seed = np.array([np.interp(u, np.linspace(0, 1, len(solved)), solved[:, j]) for j in range(14)])
            seed[:7] = start[:7]
            position = xyz.copy()
            if not reshape:
                position[0] = start_pos[0, 1, 0]+u*(xyz[0]-start_pos[0, 1, 0])
            # Allow a boundary start to move inward continuously; do not
            # suddenly impose the 2-degree IK bound on its first waypoint.
            reserve = 0. if reshape and u < 1 else self.p.goal_reserve_deg
            points.append(self.project(seed, position, quat, reserve=reserve))
        points = np.asarray(points)
        if np.max(np.abs(points[-1]-goal_joint)) > .002:
            raise Rejected('Projection changed the chosen joint branch endpoint')
        dt = max(.02, nominal_dt*(len(solved)-1)/len(points))
        for _ in range(6):
            self.check()
            samples = self.a.sample(start, points, dt, self.p.sample_dt)
            ratios = self.a.dynamics(samples, self.p.sample_dt)
            if not np.isfinite(ratios).all():
                raise Rejected('Invalid dynamics metrics')
            if np.all(ratios <= 1):
                metrics = self.metrics(start, samples, xyz, quat, reshape=reshape)
                metrics.update(dynamics_ratios=ratios.tolist(), waypoint_count=len(points), dt=dt)
                return dict(start=start.copy(), points=points, dt=dt, metrics=metrics,
                            xyz=xyz.copy(), quat=quat.copy(), reshape=reshape)
            dt *= max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))*1.1
        raise Rejected('RT interpolation dynamics failed after retiming')

    def advance(self, start, xyz, quat):
        endpoint = self.project(start, xyz, quat)
        return self.path(start, endpoint, xyz, quat)

    def solve(self, start, xyz, quat, *, allow_reshape=True):
        self.check()
        if start.shape != (14,) or xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[start, xyz, quat]).all():
            raise Rejected('Invalid preflight snapshot')
        pos, rot = self.a.tcp(start)
        if (not 0 <= xyz[0]-pos[0, 1, 0] <= .5 or
                np.linalg.norm(xyz[1:]-pos[0, 1, 1:]) > .003 or angle(rot[0, 1], quat) > .5):
            raise Rejected('Frozen pose/500mm advance contract mismatch')
        try:
            return dict(advance=self.advance(start, xyz, quat), reshape=None)
        except Rejected as error:
            self.rejections.append(dict(stage='current', reason=str(error),details=error.details))
        if not allow_reshape:
            raise Rejected('Post-reshape path failed; no second reshape permitted')
        candidates = []
        for i, seed in enumerate(seeds(start, self.a.bounds, self.p.seeds, self.p.goal_reserve_deg)):
            if candidates and self.clock() >= self.deadline-.5:
                break
            self.check()
            try:
                q = self.project(seed, pos[0, 1], quat)
                if np.max(np.abs(q-start)) < .002:
                    continue
                reshape = self.path(start, q, pos[0, 1], quat, reshape=True)
                advance = self.advance(reshape['points'][-1], xyz, quat)
                score = (min(reshape['metrics']['min_path_margin_deg'], advance['metrics']['min_path_margin_deg']),
                         -float(np.linalg.norm(q-start)))
                candidates.append((score, dict(reshape=reshape, advance=advance)))
            except Rejected as error:
                self.rejections.append(dict(stage='candidate', seed=i, reason=str(error),details=error.details))
        if not candidates:
            raise Rejected('No verified reshape plus advance path')
        return max(candidates, key=lambda item:item[0])[1]
