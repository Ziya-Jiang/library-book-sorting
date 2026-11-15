"""Opt-in post-lift TCP sequence; retain the historical J7 policy separately."""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from book_spine_wrist import _command_right_j7


def _rotation_wxyz(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > .001:
        raise ValueError('Invalid post-lift TCP orientation')
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def _fresh_right_pose(robot):
    state = robot.get_current_state()
    if not isinstance(state, dict):
        raise ValueError('No feedback for post-lift TCP motion')
    now = time.monotonic()
    for key in ('received_monotonic', 'pose_sample_monotonic'):
        stamp = state.get(key)
        if not isinstance(stamp, (int, float)) or not np.isfinite(stamp) or not 0 <= now-stamp <= .25:
            raise ValueError('Post-lift TCP feedback expired: '+key)
    pose = state.get('right_arm', {})
    xyz = np.asarray(pose.get('position'), float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('Invalid post-lift TCP position')
    return xyz*1000., _rotation_wxyz(pose.get('orientation'))


def capture_neutral_tcp(robot):
    """Capture this pick after first_reset, before J7/visual alignment."""
    _, orientation = _fresh_right_pose(robot)
    return dict(neutral_tcp_wxyz=orientation.as_quat()[[3, 0, 1, 2]].tolist(),
                neutral_tcp_source='measured_before_this_pick_prealignment')


def offset_tcp_ccw(neutral_wxyz, ccw_deg):
    """Signed CCW angle as seen from robot towards book; negative is clockwise.

    Fingers point along tool -Z. Positive local +Z rotation is CCW when
    looking along that approach direction. Do not add to a joint or Euler Y.
    """
    if isinstance(ccw_deg, bool) or not np.isfinite(ccw_deg) or not -180 <= ccw_deg <= 180:
        raise ValueError('Invalid signed post-lift TCP angle (-180 to 180 degrees)')
    return _rotation_wxyz(neutral_wxyz) * Rotation.from_euler('z', ccw_deg, degrees=True)


def hold_with_tcp_restore_and_offset(robot, hold_seconds, ccw_deg, record, *, log=None):
    """Restore at the lifted position, then optionally offset; IK chooses joints."""
    if not np.isfinite(hold_seconds) or hold_seconds < 0:
        raise ValueError('Invalid post-lift hold duration')
    neutral = _rotation_wxyz(record.get('neutral_tcp_wxyz'))
    rotated = offset_tcp_ccw(record['neutral_tcp_wxyz'], ccw_deg)
    if record.get('post_lift_started'):
        raise ValueError('Post-lift TCP sequence was already attempted for this pick')

    def report(event):
        print('抓取抬升后末端姿态：'+json.dumps(event, ensure_ascii=False), flush=True)
        if log is not None:
            log(event)

    anchor, _ = _fresh_right_pose(robot)
    started = time.monotonic()
    record.update(post_lift_started=True, post_lift_complete=False,
                  post_lift_tcp_ccw_deg=float(ccw_deg), post_lift_tcp_anchor_mm=anchor.tolist())
    # Keep the existing action identity; its goal accepts signed CCW angles.
    stages = [('restore', neutral)]
    if ccw_deg != 0:
        stages.append(('ccw', rotated))
    for stage, goal in stages:
        getattr(robot, 'operator_checkpoint', lambda: None)()
        before_xyz, before = _fresh_right_pose(robot)
        quaternion = goal.as_quat()[[3, 0, 1, 2]].tolist()
        action = 'book_post_lift_pose_'+stage
        report(dict(event='requested', stage=stage, target_xyz_mm=anchor.tolist(),
                    target_wxyz=quaternion, measured_xyz_mm=before_xyz.tolist(),
                    measured_wxyz=before.as_quat()[[3, 0, 1, 2]].tolist(),
                    requested_rotation_from_current_deg=float(np.rad2deg((before.inv()*goal).magnitude())),
                    ccw_from_neutral_deg=0. if stage == 'restore' else float(ccw_deg),
                    rotation_direction=('restore' if stage == 'restore' else
                        'clockwise' if ccw_deg < 0 else 'counterclockwise' if ccw_deg > 0 else 'none'),
                    rotation_degrees=0. if stage == 'restore' else abs(float(ccw_deg)),
                    reference_source=record.get('neutral_tcp_source')))
        pose = np.r_[anchor, goal.as_euler('xyz', degrees=True)].tolist()
        if robot.master_control(right_arm=pose, action=action, object_category='book') is not True:
            report(dict(event='not_confirmed', stage=stage, action=action))
            return False
        actual_xyz, actual = _fresh_right_pose(robot)
        position_error = float(np.linalg.norm(actual_xyz-anchor))
        angle_error = float(np.rad2deg((goal.inv()*actual).magnitude()))
        relative = neutral.inv()*actual
        matrix = relative.as_matrix()
        actual_ccw = float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))
        confirmed = position_error <= 3. and angle_error <= 1.
        report(dict(event='confirmed' if confirmed else 'not_confirmed', stage=stage,
                    actual_xyz_mm=actual_xyz.tolist(), actual_wxyz=actual.as_quat()[[3, 0, 1, 2]].tolist(),
                    position_error_mm=position_error, orientation_error_deg=angle_error,
                    measured_rotation_from_neutral_deg=float(np.rad2deg(relative.magnitude())),
                    measured_ccw_from_neutral_deg=actual_ccw))
        if not confirmed:
            return False
    record['post_lift_target_wxyz'] = rotated.as_quat()[[3, 0, 1, 2]].tolist()
    remaining = max(0., hold_seconds-(time.monotonic()-started))
    if remaining:
        getattr(robot, 'operator_sleep', time.sleep)(remaining)
    record['post_lift_complete'] = True
    return True


def read_first_reset_j7(robot):
    """Use the same configured right-arm initial angle as first_reset."""
    ready = np.asarray(getattr(robot, 'right_safe_joint_deg', None), float)
    if ready.shape != (7,) or not np.isfinite(ready).all():
        raise ValueError('Invalid first_reset right-arm joint configuration')
    return float(ready[6])


def hold_with_j7_reset_and_offset(robot, hold_seconds, offset_deg, record,
                                  lower_deg, upper_deg, margin_deg=5., *, log=None):
    """Restore first_reset J7, then command first_reset J7 + signed offset.

    The offset is in joint degrees, with no viewpoint-dependent sign change.
    Other joint targets come from measured state at each step. This is joint
    planning, not a claim of a fixed Cartesian TCP path or whole-arm reset.
    """
    values = np.asarray([hold_seconds, offset_deg, lower_deg, upper_deg, margin_deg], float)
    if (not np.isfinite(values).all() or hold_seconds < 0
            or margin_deg < 0 or lower_deg+margin_deg > upper_deg-margin_deg):
        raise ValueError('Invalid post-lift J7/hold configuration')
    reference = read_first_reset_j7(robot)
    targets = [('restore', reference, 'book_post_lift_j7_restore'),
               ('offset', reference+float(offset_deg), 'book_post_lift_j7_offset')]
    if any(not lower_deg+margin_deg <= target <= upper_deg-margin_deg
           for _, target, _ in targets):
        raise ValueError('Post-lift J7 target violates the configured soft-limit margin')
    if record.get('post_lift_started'):
        raise ValueError('Post-lift J7 sequence was already attempted for this pick')

    def report(event):
        print('抓取抬升后J7：'+json.dumps(event, ensure_ascii=False), flush=True)
        if log is not None:
            log(event)

    started = time.monotonic()
    record['first_reset_j7_deg'] = reference
    record['post_lift_j7_offset_deg'] = float(offset_deg)
    record['post_lift_started'] = True
    record['post_lift_complete'] = False
    for stage, target, action in targets:
        getattr(robot, 'operator_checkpoint', lambda: None)()
        state = robot.get_current_state()
        if state is None or 'joint_state' not in state:
            raise ValueError('No joint feedback for post-lift J7 '+stage)
        joints = np.asarray(state['joint_state'], float).reshape(2, 7)
        if not np.isfinite(joints).all():
            raise ValueError('Invalid joint feedback for post-lift J7 '+stage)
        report(dict(event='requested', stage=stage,
                    measured_j7_deg=float(np.rad2deg(joints[1, 6])),
                    first_reset_j7_deg=reference, offset_deg=float(offset_deg),
                    target_j7_deg=target, reference_source='first_reset.right_safe_joint_deg[6]'))
        step_record = {}
        if not _command_right_j7(robot, state, target, action, step_record):
            report(dict(event='not_confirmed', stage=stage, target_j7_deg=target))
            return False
        record[stage+'_j7_deg'] = step_record['after_deg']
        report(dict(event='confirmed', stage=stage, target_j7_deg=target,
                    measured_j7_deg=step_record['after_deg']))
    remaining = max(0., hold_seconds-(time.monotonic()-started))
    if remaining:
        getattr(robot, 'operator_sleep', time.sleep)(remaining)
    record['post_lift_complete'] = True
    return True
