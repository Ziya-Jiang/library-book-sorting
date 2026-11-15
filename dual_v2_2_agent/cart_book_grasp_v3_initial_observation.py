"""Bounded stationary acquisition; no motion commands or model calls here."""
import numpy as np
from cart_book_grasp_v2_dual_view import ObservationMoved
from cart_book_grasp_v3_servo import ServoError


def stationary_observation(robot,operation,*,checkpoint,sleep,clock,log,deadline,
                           tolerance_deg=.5,attempts=3):
    def check():
        checkpoint()
        if clock()>=deadline:raise ServoError('Initial wrist observation total time budget exceeded')
    def sample():
        check();s=robot.get_current_state();now=clock()
        if not s:raise ServoError('Initial wrist observation missing robot feedback')
        stamp=float(s.get('received_monotonic',float('nan')))
        q=np.asarray(s.get('joint_state'),float)
        status=getattr(s.get('curobo_status'),'value',s.get('curobo_status'))
        if status in (3,7,8):raise ServoError(f'Initial wrist observation backend fault: status={status}')
        if q.size!=14 or not np.isfinite(q).all() or not np.isfinite(stamp) or not 0<=now-stamp<=.25:
            raise ServoError('Initial wrist observation invalid/stale feedback')
        return stamp,q.reshape(14)
    for attempt in range(1,attempts+1):
        end=min(deadline,clock()+3.);window=[];last=-np.inf
        log(dict(event='wrist_observation_settling',attempt=attempt,timeout_seconds=3.))
        while True:
            stamp,q=sample()
            if stamp>last:
                last=stamp;window.append((stamp,q.copy()))
                # Keep the window since the most recent excessive movement.
                while len(window)>1 and np.degrees(np.ptp(np.array([x[1] for x in window]),axis=0)).max()>min(.1,tolerance_deg/2):
                    window.pop(0)
                if len(window)>=3 and stamp-window[0][0]>=.3:break
            if clock()>=end:raise ServoError('Initial wrist observation did not settle within 3s')
            sleep(.02)
        try:
            result=operation()
            sample();check()
            log(dict(event='wrist_observation_ready',attempt=attempt))
            return result
        except ObservationMoved as error:
            log(dict(event='wrist_observation_retry',attempt=attempt,reason=str(error),
                     policy='discard_selection_wait_stable_recapture'))
            if attempt==attempts:
                raise ServoError(f'Initial wrist observation moved in all {attempts} attempts: {error}') from error
