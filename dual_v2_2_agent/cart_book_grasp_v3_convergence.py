"""Detect sustained visual divergence from distinct fresh observations."""
from collections import deque
import math


class VisualConvergence:
    def __init__(self, tolerance, stall_seconds=8.):
        self.tolerance=tuple(float(x) for x in tolerance);self.stall_seconds=stall_seconds
        self.active=False;self.reset()

    def reset(self):
        self.samples=deque(maxlen=16);self.last_stamp=None
        self.best_score=None;self.best_x=None;self.progress_at=None

    def observe(self, x, error, stamp):
        if not self.active:return None
        values=[x,stamp,*error]
        if not all(math.isfinite(v) for v in values):raise ValueError('Invalid convergence observation')
        if self.last_stamp is not None and stamp<=self.last_stamp:return None
        self.last_stamp=stamp
        score=max(abs(float(v))/t for v,t in zip(error,self.tolerance))
        if self.progress_at is None:
            self.best_score=score;self.best_x=x;self.progress_at=stamp
        # Improvement is relative to the best achieved error/forward position,
        # never arbitrary sideways travel or oscillation.
        if score<self.best_score-.5 or (x>self.best_x+1. and score<=1.5):
            self.best_score=min(self.best_score,score);self.best_x=max(self.best_x,x);self.progress_at=stamp
        self.best_score=min(self.best_score,score)
        self.samples.append((stamp,score))
        if score<=1.5:
            self.samples.clear()
            return None
        recent=[s for s in self.samples if stamp-s[0]<=1.6]
        if len(recent)>=5 and stamp-recent[0][0]>=1.:
            first=sum(v for _,v in recent[:3])/3.;last=sum(v for _,v in recent[-3:])/3.
            if last>max(first+2.,first*1.5):
                return f'Visual correction diverging: normalized_error={first:.2f}->{last:.2f}; hold and reacquire'
        if stamp-self.progress_at>self.stall_seconds:
            return f'Visual correction has no error/forward progress for {self.stall_seconds:g}s; hold and reacquire'
        return None
