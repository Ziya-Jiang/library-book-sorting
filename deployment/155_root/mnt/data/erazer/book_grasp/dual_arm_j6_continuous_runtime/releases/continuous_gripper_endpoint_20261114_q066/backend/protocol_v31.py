"""Preserve the installed RT codec while forwarding V3.1 native mode/time/ratio."""
import math
from .protocol import DEFAULT_GRIPPER_JOINT_SIZES, unpack_state
from .protocol import pack_command as legacy_pack_command


def pack_command(*args, planner_mode=0, planner_ratio=.1, **kwargs):
    if planner_mode not in (0, 1, 2) or not math.isfinite(planner_ratio):
        raise ValueError('Invalid V3.1 RT mode')
    if not 0 < planner_ratio <= (.3 if planner_mode == 2 else .1):
        raise ValueError('Invalid V3.1 RT ratio')
    raw = legacy_pack_command(*args, **kwargs)
    fields = raw.decode('ascii').rstrip('#').split('#')
    if planner_mode == 2:
        if not args[4] or kwargs.get('waypoints') or float(fields[12]) != 0:
            raise ValueError('RAW requires active zero-time single point')
        fields[19] = '2'
    fields[11] = str(planner_ratio)
    return ('#'.join(fields)+'#').encode('ascii')
