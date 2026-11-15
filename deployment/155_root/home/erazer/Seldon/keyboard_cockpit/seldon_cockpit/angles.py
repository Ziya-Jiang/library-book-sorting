"""Read-only joint angle telemetry; no publishers or actuator services."""
import math

AXES = ('steer_fr', 'propel_fr', 'steer_rl', 'propel_rl')


class AngleFeedback:
    def __init__(self):
        self.positions = {}
        self.geometry = {}
        self.geometry_at = -1e9

    def position(self, axis, value, stamp):
        self.positions[axis] = (value if value is not None and math.isfinite(value) else None, stamp)

    def calibration(self, names, offsets, xs, ys, now):
        self.geometry = {}
        if not (len(names) == len(offsets) == len(xs) == len(ys)):
            return
        for name, offset, x, y in zip(names, offsets, xs, ys):
            axis = name.rsplit('/', 1)[-1]
            if axis in ('steer_fr', 'steer_rl') and all(math.isfinite(v) for v in (offset, x, y)):
                # The existing chassis wrapper leaves steer_direction_sign at +1.
                self.geometry[axis] = dict(offset=offset, x=x, y=y)
        self.geometry_at = now

    def snapshot(self, now):
        joints = {}
        for axis in AXES:
            value, stamp = self.positions.get(axis, (None, -1e9))
            age = max(0., now-stamp) if stamp > 0 else None
            fresh = value is not None and 0 <= now-stamp <= .5
            joints[axis] = dict(rad=value if fresh else None,
                                deg=math.degrees(value) if fresh else None,
                                age=age, fresh=fresh)
        wheels = {key: dict(heading_deg=None, fresh=False, reason='未接入角度反馈')
                  for key in ('fl', 'fr', 'rl', 'rr')}
        for key, axis in [('fr', 'steer_fr'), ('rl', 'steer_rl')]:
            joint, geometry = joints[axis], self.geometry.get(axis)
            valid = joint['fresh'] and geometry is not None and 0 <= now-self.geometry_at <= 5.
            heading = None
            if valid:
                heading = math.degrees((joint['rad']-geometry['offset']+math.pi) % (2*math.pi)-math.pi)
            wheels[key] = dict(heading_deg=heading, fresh=valid,
                               reason='' if valid else '等待反馈或零位配置',
                               axis=axis, age=joint['age'])
        return dict(joints=joints, wheels=wheels, calibration_age=max(0., now-self.geometry_at))
