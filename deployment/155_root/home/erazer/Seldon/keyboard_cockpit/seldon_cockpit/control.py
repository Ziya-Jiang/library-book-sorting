"""Pure control logic. Units are m/s, rad/s, and monotonic seconds."""
import math


def validate_config(c):
    for key in ("linear_acceleration", "linear_deceleration", "angular_acceleration",
                "angular_deceleration", "heartbeat_timeout", "camera_timeout", "chassis_camera_timeout",
                "network_lock_timeout", "key_repeat_timeout", "max_hold_seconds", "initialization_speed_tolerance",
                "initialization_speed_hard_limit", "initialization_speed_grace",
                "zero_hold_seconds", "service_timeout"):
        if not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    if c["heartbeat_timeout"] > 0.5 or c["zero_hold_seconds"] < 0.6:
        raise ValueError("heartbeat_timeout <= 0.5 and zero_hold_seconds >= 0.6 required")
    if not c["heartbeat_timeout"] < c["network_lock_timeout"] <= 2.0:
        raise ValueError("heartbeat_timeout < network_lock_timeout <= 2.0 required")
    if c["initialization_speed_tolerance"] >= c["initialization_speed_hard_limit"]:
        raise ValueError("initialization speed tolerance must be below hard limit")
    previous = (0, 0)
    for name in ("slow", "normal", "fast"):
        v, w = c["speeds"][name]
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (v, w)):
            raise ValueError("Speed must be finite")
        if not (previous[0] <= v <= 1.5 and previous[1] <= w <= 1.5 and v > 0 and w > 0):
            raise ValueError("Speeds must be ordered, positive and <= chassis 1.5 limits")
        previous = v, w
    return c


def target(keys, speeds):
    keys = set(keys)
    gear = "slow" if "space" in keys else "fast" if "shift" in keys else "normal"
    v, w = speeds[gear]
    x = int("up" in keys or "w" in keys) - int("down" in keys or "s" in keys)
    y = int("a" in keys) - int("d" in keys)
    yaw = int("left" in keys) - int("right" in keys)
    norm = max(1.0, math.hypot(x, y))
    return (v*x/norm, v*y/norm, w*yaw), gear




class Motion:
    def __init__(self, config):
        self.config = config
        self.velocity = (0.0, 0.0, 0.0)

    def stop(self):
        self.velocity = (0.0, 0.0, 0.0)
        return self.velocity

    def step(self, desired, dt):
        dt = max(0.0, min(dt, 0.04))
        x, y, w = self.velocity
        tx, ty, tw = desired
        slowing = math.hypot(tx, ty) < math.hypot(x, y) or x*tx+y*ty < 0
        rate = self.config["linear_deceleration" if slowing else "linear_acceleration"]
        distance = math.hypot(tx-x, ty-y)
        scale = min(1.0, rate*dt/distance) if distance else 1.0
        arate = self.config["angular_deceleration" if abs(tw) < abs(w) or tw*w < 0 else "angular_acceleration"]
        self.velocity = (x+(tx-x)*scale, y+(ty-y)*scale, w+max(-arate*dt, min(arate*dt, tw-w)))
        return self.velocity


def stop_reason(intent, now, config, real):
    if not intent or not 0 <= now-intent["time"] <= config["network_lock_timeout"]:
        return "窗口心跳中断"
    if not intent.get("focused") and intent.get("keys"):
        return "窗口失焦且运动键未释放"
    if intent.get("stop"):
        return intent.get("stop_reason") or "J 停车 / 已锁定"
    # During a short network pause the controller already forces zero. Do not
    # misclassify the same delayed packet as a camera failure.
    if (real and now-intent["time"] <= config["heartbeat_timeout"]
            and not 0 <= now-intent.get("frame_time", 0) <= config["camera_timeout"]):
        return "相机画面中断"
    return ""
