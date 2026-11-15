"""Explicit right-only hardware policy; inactive left is a static model reference."""
import math


def right_only(controller):
    return getattr(controller, 'rt_arm_mode', 'both') == 'right'


def required_joints_enabled(controller):
    arms = controller.state['arms']
    if len(arms) != 2 or any(len(a['enabled']) != 7 for a in arms):
        return False
    if not right_only(controller):
        return all(all(a['enabled']) for a in arms)
    # Disabled left must actually be reported disabled by the matching RT.
    return not any(arms[0]['enabled']) and all(arms[1]['enabled'])


def validate_static_left(controller, raw_points):
    if not right_only(controller):
        return
    left = controller.state['arms'][0]['position']
    for point in raw_points:
        if len(point[0]) != 7 or any(not math.isfinite(float(q)) or
                abs(float(q)-float(anchor)) > 1e-4 for q, anchor in zip(point[0], left)):
            raise ValueError('Right-only mode rejects left-arm motion; left pose is a static reference')


def validate_manual_action(controller, message):
    if not right_only(controller):
        return
    # Right-only entry is for the checked Planner path. Do not let generic
    # manual reset/cartesian actions synthesize a two-arm target.
    if message.get('action') in ('claim', 'cartesian', 'joint', 'reset', 'zero', 'initial_pose'):
        raise ValueError('Right-only mode: manual arm control unavailable; use right-arm Planner')
