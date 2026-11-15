"""Small Cartesian jogs in the upper-body frame, retaining wrist orientation.

This is a local differential IK solver, not a collision-aware planner. Targets
are seeded from measured V2 joint angles on every update, never a home pose.
"""
import math
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)


class Kinematics:
    def __init__(self, manifest):
        self.names = manifest['arm_joints']
        joints = {j['name']: j for j in manifest['joints']}
        parents = {j['child']: j for j in manifest['joints']}
        self.chains = []
        for a, side in enumerate(('l', 'r')):
            root = joints[self.names[a][0]]['parent']
            chain, link = [], 'arm_end_effector_'+side
            while link != root:
                joint = parents[link]
                if joint['type'] != 'fixed' and joint['name'] not in self.names[a]:
                    raise ValueError('未跟踪的关节进入双臂 IK 链')
                chain.append(joint)
                link = joint['parent']
            # Include fixed shoulder mounting transforms, but not an untracked waist.
            while root in parents and parents[root]['type'] == 'fixed':
                chain.append(parents[root])
                root = parents[root]['parent']
            self.chains.append(list(reversed(chain)))

    def pose(self, arm, q, jacobian=False):
        values = dict(zip(self.names[arm], q))
        transform = np.eye(4)
        pivots = []
        for joint in self.chains[arm]:
            r, p, y = joint['rpy']
            origin = np.eye(4)
            origin[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            origin[:3, 3] = joint['xyz']
            transform = transform @ origin
            if joint['name'] in values:
                axis = np.asarray(joint['axis'], float)
                axis /= np.linalg.norm(axis)
                pivots.append((transform[:3, 3].copy(), transform[:3, :3] @ axis))
                motion = np.eye(4)
                motion[:3, :3] = rotation(axis, values[joint['name']])
                transform = transform @ motion
        if not jacobian:
            return transform
        position = transform[:3, 3]
        return transform, np.column_stack([np.r_[np.cross(axis, position-pivot), axis] for pivot, axis in pivots])

    def jog(self, arm, q, delta, bounds):
        q, delta = np.asarray(q, float), np.asarray(delta, float)
        if q.shape != (7,) or delta.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(delta).all():
            raise ValueError('无效的末端增量')
        if np.linalg.norm(delta) > .010001:
            raise ValueError('单次末端增量不得超过 10 mm')
        before, jac = self.pose(arm, q, True)
        # Angular rows constrain wrist rotation; damping softens singular poses.
        error = np.r_[delta, [0., 0., 0.]]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .025**2*np.eye(6), error)
        dq *= min(1., math.radians(1.) / max(np.max(np.abs(dq)), 1e-12))
        target = q + dq
        if any(not lo <= value <= hi for value, (lo, hi) in zip(target, bounds)):
            raise ValueError('末端运动触及关节限位')
        after = self.pose(arm, target)
        achieved = after[:3, 3]-before[:3, 3]
        if np.linalg.norm(delta) > 1e-8 and (np.dot(achieved, delta) <= 0 or np.linalg.norm(achieved) < .1*np.linalg.norm(delta)):
            raise ValueError('该方向接近奇异位姿，无法可靠移动')
        return target.tolist()
