"""Wire contract of the installed C++ CuarmMessageHandler (not the older Python API).
All joint position/velocity values on this wire are radians / radians per second.
"""
import math

NAMES = ['armL-DualV2_2Arm', 'armR-DualV2_2Arm']
# RT assembly order: left Inspire hand (6 joints), right UART two-finger
# gripper (1 joint).  The wire protocol carries a size for every gripper, so
# the parser remains compatible with a temporary no-gripper simulator.
DEFAULT_GRIPPER_JOINT_SIZES = (1, 1)

class Reader:
    def __init__(self, raw):
        text = raw.decode('ascii').rstrip('\x00')
        if not text.endswith('#') or len(text) > 60000:
            raise ValueError('Incomplete CUARM packet')
        self.fields = iter(text[:-1].split('#'))
    def take(self, cast=float):
        try:
            value = cast(next(self.fields))
        except (StopIteration, ValueError) as exc:
            raise ValueError('Malformed CUARM packet') from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite CUARM value')
        return value
    def array(self, n, cast=float):
        return [self.take(cast) for _ in range(n)]
    def end(self):
        if next(self.fields, None) is not None:
            raise ValueError('Unexpected CUARM protocol fields')

def unpack_state(raw):
    r = Reader(raw)
    state = dict(echo=r.take(int), timestamp=r.take(int), ready=bool(r.take(int)), sequence=r.take(int))
    if r.take(int) != 2:
        raise ValueError('Expected dual arm')
    arms = []
    for _ in range(2):
        if r.take(int) != 7:
            raise ValueError('Expected seven joints per arm')
        arm = {'enabled': r.array(7, int)}
        for key in ('position', 'velocity', 'torque', 'motor_position', 'motor_velocity', 'motor_current', 'motor_torque'):
            arm[key] = r.array(7)
        arm['joint_pose'] = [r.array(7) for _ in range(7)]
        arm['ee_pose'], arm['tool_pose'], arm['target'] = r.array(7), r.array(7), r.array(7)
        arms.append(arm)
    gripper_count = r.take(int)
    if not 0 <= gripper_count <= 2:
        raise ValueError('Invalid gripper count')
    grippers = []
    for _ in range(gripper_count):
        joint_size = r.take(int)
        if not 1 <= joint_size <= 7:
            raise ValueError('Invalid gripper joint count')
        grippers.append({'joint_size': joint_size, 'position': r.array(joint_size)})
    for arm in arms:
        arm['tool_offset'] = r.array(3)
    for arm in arms:
        for key in ('soft_position', 'soft_velocity', 'soft_torque', 'hard_position', 'hard_velocity', 'hard_torque'):
            arm[key] = [r.array(2) for _ in range(7)]
        for key in ('follow_position', 'follow_velocity', 'follow_torque', 'jump_position', 'jump_velocity', 'jump_torque'):
            arm[key] = r.array(7)
    for gripper in grippers:
        size = gripper['joint_size']
        gripper['soft_position'] = [r.array(2) for _ in range(size)]
        gripper['hard_position'] = [r.array(2) for _ in range(size)]
    state.update(system=r.take(int), plan_result=r.take(int), diagnostics=r.take(int), arms=arms, grippers=grippers)
    for arm in arms:
        arm['diagnostics'] = r.array(7, int)
    r.end()
    return state

def pack_command(timestamp, sequence, state, target, active, simulation, duration=5.0,
                 gripper_targets=None, gripper_joint_sizes=DEFAULT_GRIPPER_JOINT_SIZES,
                 waypoints=None, waypoint_dt=.5):
    # connection Remote means panel/RT connection, independent of IP selection.
    f = [timestamp, sequence, 1, int(not active), int(simulation), 0, 0, 0,
         0, 0, 1, 0.10, duration, 0, 0.10, 0, 0, 0, 0, 0, 2]
    if waypoints:
        if not active or not 2 <= len(waypoints) <= 64 or not .05 <= waypoint_dt <= 5:
            raise ValueError('Invalid active trajectory')
        f[19] = 1  # native PanelTargetMode::kWaypoint
    for a in range(2):
        arm = state['arms'][a] if state else None
        f += [7, NAMES[a]]
        # Preserve driver's current enable state. Opening the web page never enables a joint.
        f += arm['enabled'] if arm else [0]*7
        f += arm['motor_position'] if arm else [0]*7
        f += target[a]
        f += [0]*6
    f += [len(waypoints or []), waypoint_dt if waypoints else .01]
    for point in waypoints or []:
        if len(point) != 2 or any(len(row) != 7 for row in point):
            raise ValueError('Invalid waypoint shape')
        if any(not math.isfinite(x) for row in point for x in row):
            raise ValueError('Nonfinite waypoint')
        f += point[0] + point[1] + [waypoint_dt, 0.]
    sizes = tuple(int(size) for size in gripper_joint_sizes)
    if len(sizes) > 2 or any(size < 1 or size > 7 for size in sizes):
        raise ValueError('Invalid gripper joint count')
    if gripper_targets is None:
        gripper_targets = ([g['position'][:] for g in state.get('grippers', [])]
                           if state else [[0.0] * size for size in sizes])
    if len(gripper_targets) != len(sizes):
        raise ValueError('Gripper target count does not match protocol')
    f += [len(sizes)]
    for size, values in zip(sizes, gripper_targets):
        if len(values) != size or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in values):
            raise ValueError('Invalid gripper target')
        f += [size] + list(values)
    for a in range(2):
        f += state['arms'][a]['tool_offset'] if state else [0]*3
    for a in range(2):
        for key in ('soft_position', 'soft_velocity'):
            for pair in (state['arms'][a][key] if state else [[0, 0]]*7):
                f += pair
    for index, size in enumerate(sizes):
        limits = ([[0.0, 1.0]] * size)
        if state and len(state.get('grippers', [])) == len(sizes):
            limits = state['grippers'][index].get('soft_position', limits)
        if len(limits) != size or any(len(pair) != 2 for pair in limits):
            raise ValueError('Invalid gripper limits')
        for pair in limits:
            f += pair
    return ('#'.join(str(x) for x in f) + '#').encode('ascii')
