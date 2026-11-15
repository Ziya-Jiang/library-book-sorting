"""Bounded visual completion handoff: acknowledge a real hold, never run IK."""
import time
import copy
import math
import numpy as np


def pose_close(a, b, position=.003, angle_deg=1.):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=(2,7) or b.shape!=(2,7) or not np.isfinite([a,b]).all():return False
    na=np.linalg.norm(a[:,3:],axis=1);nb=np.linalg.norm(b[:,3:],axis=1)
    if np.any(na<1e-6) or np.any(nb<1e-6):return False
    cosine=np.abs(np.sum(a[:,3:]*b[:,3:],axis=1)/(na*nb))
    return bool(np.all(np.linalg.norm(a[:,:3]-b[:,:3],axis=1)<=position)
        and np.all(2*np.arccos(np.clip(cosine,0,1))<=np.deg2rad(angle_deg)))


class MeasuredHandoff:
    def __init__(self, state, expected, session, previous_id, now):
        pose=np.c_[state['position'],state['orientation']]
        if not pose_close(pose,expected):
            raise ValueError('Visual completion is outside arrival tolerance; handoff rejected')
        self.pose=pose.tolist();self.session=session;self.previous_id=previous_id
        self.deadline=now+2.;self.accepted=None;self.identity=None

    def accept(self, command, session, state, now):
        if session!=self.session:raise ValueError('Visual handoff session changed')
        if self.accepted is None and now>self.deadline:
            raise ValueError('Visual hold handoff timed out; no Cartesian correction sent')
        key=command.get('target_id')
        if (command.get('action')!='book_v3_1_stream' or type(key) is not int
                or key<=self.previous_id or not pose_close(command.get('pose'),self.pose,1e-8,1e-5)):
            raise ValueError('Visual handoff target changed; explicit new task required')
        current=np.c_[state['position'],state['orientation']]
        if not pose_close(current,self.pose,.001,.5):
            raise ValueError('Arm moved during visual hold handoff')
        if self.accepted is not None:
            if key < self.accepted or (key == self.accepted and command != self.identity):
                raise ValueError('Visual handoff identity changed')
            if key > self.accepted and command['target_echo'] < self.identity['target_echo']:
                raise ValueError('Visual handoff timestamp moved backwards')
        # A new receipt/ID may carry a gripper update while both arms retain
        # this exact completed pose. It is still a measured hold, never IK.
        self.accepted=key;self.identity=copy.deepcopy(command)
        return True

    def status(self):
        return dict(active=True,target_id=self.accepted,holding=True,paused=False,fault=None)


def validated_grippers(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Invalid visual hold gripper shape')
    result=[]
    for row, size in zip(value, (1, 1)):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError('Invalid visual hold gripper shape')
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in row):
            raise ValueError('Invalid visual hold gripper target')
        result.append(list(row))
    return result
