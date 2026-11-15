"""Read-only pre-arm settling; thresholds match the backend's raw RT units."""
import math

VELOCITY_LIMIT = .02
STABLE_SECONDS = .3
TIMEOUT_SECONDS = 8.


class StationaryWindow:
    def __init__(self):
        self.reset()
        self.last_source = None
        self.last_seen = None
        self.latest = None
        self.peak = None

    def reset(self):
        self.start = self.source_start = None
        self.samples = 0

    def observe(self, state, now):
        if not state.get('fresh'):
            raise RuntimeError('等待停稳期间RT反馈已过期；未使能Planner')
        raw = state.get('state') or {}
        source = raw.get('timestamp')  # RT microseconds, not the web challenge.
        arms = raw.get('arms')
        gain = state.get('calibration', {}).get('gain')
        try:
            if (not isinstance(source, (int, float)) or isinstance(source, bool)
                    or not math.isfinite(source) or source <= 0
                    or len(arms) != 2 or len(gain) != 2):
                raise ValueError()
            values = []
            for a in range(2):
                if len(arms[a]['velocity']) != 7 or len(gain[a]) != 7:
                    raise ValueError()
                for j in range(7):
                    v, g = float(arms[a]['velocity'][j]), float(gain[a][j])
                    if not math.isfinite(v) or not math.isfinite(g) or g == 0:
                        raise ValueError()
                    values.append((abs(v/g), a, j, v/g))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            raise RuntimeError('停稳检查缺少有效RT时间戳、双臂速度或标定；未使能Planner') from None
        if self.last_source is not None and source <= self.last_source:
            if source < self.last_source:
                raise RuntimeError('等待停稳期间RT时间戳倒退；未使能Planner')
            if now-self.last_seen > .2:
                self.reset()
            return False
        if self.last_seen is not None and (now-self.last_seen > .2 or source-self.last_source > 200000):
            self.reset()
        self.last_source, self.last_seen = source, now
        speed, a, j, signed = max(values)
        self.latest = dict(arm=('left', 'right')[a], joint=j+1, velocity_rad_s=signed,
                           abs_velocity_rad_s=speed, limit_rad_s=VELOCITY_LIMIT)
        if self.peak is None or speed > self.peak['abs_velocity_rad_s']:
            self.peak = self.latest.copy()
        if speed > VELOCITY_LIMIT:
            self.reset()
            return False
        if self.start is None:
            self.start, self.source_start = now, source
        self.samples += 1
        return (self.samples >= 5 and now-self.start >= STABLE_SECONDS
                and (source-self.source_start)*1e-6 >= STABLE_SECONDS)

    def description(self):
        def describe(v):
            if not v:return '无有效样本'
            return f"{v['arm']} J{v['joint']} {v['velocity_rad_s']:.6f}rad/s"
        return f'最新={describe(self.latest)}，峰值={describe(self.peak)}，阈值={VELOCITY_LIMIT:.3f}rad/s'
