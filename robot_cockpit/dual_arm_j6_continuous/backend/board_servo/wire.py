"""Cartesian extension. Echo uses robot clock; keepalives never renew a target."""
import math
import numpy as np

ACTIONS={'book_v3_xyz_servo','book_v3_1_stream'}


def target(packet,now,echo_age,expected_digest):
    if packet.get('board_servo_sha')!=expected_digest:raise ValueError('Board model/code mismatch')
    key=packet.get('target_id');action=packet.get('action');anchor=packet.get('target_echo')
    if type(key) is not int or key<1 or action not in ACTIONS:raise ValueError('Invalid Cartesian identity/action')
    if type(anchor) not in (int,float) or not math.isfinite(anchor) or not 0<=now-anchor<=5.:
        raise ValueError('Cartesian target timestamp invalid/expired')
    pose=np.asarray(packet.get('pose'),float)
    if pose.shape!=(2,7) or not np.isfinite(pose).all():raise ValueError('Invalid Cartesian pose')
    if np.max(np.abs(np.linalg.norm(pose[:,3:],axis=1)-1.))>1e-4:raise ValueError('Invalid Cartesian quaternion')
    return dict(target_id=key,action=action,target_echo=anchor,pose=pose.tolist())
