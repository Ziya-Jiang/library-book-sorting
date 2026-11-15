"""Stopped-arm GPT reacquisition; never control from an API-aged image."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from cart_book_grasp_v3_servo import ServoError
from cart_book_grasp_v3_feature_centerline import FeatureCenterlineLock, FeaturePointsExhausted, InitialFeaturesUnavailable, CenterlineGeometryError


class TrackingRecoveryRejected(ServoError):
    """Only semantic/feature verification is retryable; motion faults are not."""


class RecoveryProgress:
    """Bound repeated reacquisition without counting lifetime successes as failures."""
    def __init__(self, clock, log, *, failure_limit=3, stall_seconds=60.):
        self.clock,self.log=clock,log
        self.failure_limit,self.stall_seconds=failure_limit,stall_seconds
        self.failures=0;self.last_progress=None;self.anchor=None
        self.stamp=-np.inf;self.since=None;self.samples=0;self.best_x=-np.inf

    def check(self):
        if self.last_progress is not None and self.clock()-self.last_progress>=self.stall_seconds:
            raise ServoError(f'GPT recovery no effective progress for {self.stall_seconds:g}s')

    def begin(self, x):
        self.check()
        if self.last_progress is None:self.last_progress=self.clock()
        self.best_x=max(self.best_x,float(x))
        self.anchor=None;self.since=None;self.samples=0

    def rejected(self, error):
        self.failures+=1
        self.log(dict(event='gpt_tracking_recovery_rejected',consecutive_failures=self.failures,
                      failure_limit=self.failure_limit,reason=str(error)))
        if self.failures>=self.failure_limit:
            raise ServoError(f'GPT consecutive recovery failures reached {self.failure_limit}: {error}') from error
        self.check()

    def verified(self, x, error, stamp):
        # Re-selection itself is not progress, even if its reported error jumps.
        self.anchor=(float(x),np.abs(np.asarray(error,float)))
        self.stamp=float(stamp);self.since=None;self.samples=0

    def observe(self, x, error, stamp):
        self.check()
        if self.anchor is None:return
        if stamp<=self.stamp:return
        if stamp-self.stamp>.75:self.since=None;self.samples=0
        self.stamp=float(stamp)
        if self.since is None:self.since=self.clock()
        self.samples+=1
        e=np.abs(np.asarray(error,float));old=self.anchor[1]
        forward=float(x)-self.best_x>=3.
        # Meaningful improvement on the SAME recovered feature identity,
        # without trading a better pixel error for worse orientation (or vice versa).
        correction=((old[0]-e[0]>=3. and e[1]<=old[1]+.25) or
                    (old[1]-e[1]>=.5 and e[0]<=old[0]+1.))
        if self.samples>=3 and self.clock()-self.since>=.5 and (forward or correction):
            previous=self.failures;self.failures=0;self.last_progress=self.clock()
            self.best_x=max(self.best_x,float(x));self.anchor=(float(x),e)
            self.since=None;self.samples=0
            self.log(dict(event='gpt_recovery_progress_confirmed',cleared_failures=previous,
                          measured_x_mm=float(x),error=e.tolist(),kind='forward' if forward else 'alignment'))
        self.check()


def make_lock(rgb,chosen,*,factory=FeatureCenterlineLock):
    kwargs=dict(tracking_mask=chosen.get('tracking_mask'))
    if chosen.get('centerline') is not None:kwargs['centerline']=chosen['centerline']
    return factory(rgb,chosen['mask'],**kwargs)


def initialize_tracking(frame,chosen,select,capture,checkpoint,directory,*,factory=FeatureCenterlineLock,max_frame_age_seconds=None,clock=time.monotonic):
    """One GPT retry for missing initial texture, verified on a fresh image.

    capture/checkpoint belong to the stationary acquisition guard; this helper
    neither issues movement nor converts failed initialization into point loss.
    """
    try:return make_lock(frame['rgb'],chosen,factory=factory)
    except InitialFeaturesUnavailable as error:
        if select is None:raise
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        record=dict(trigger='initial_features_unavailable',reason=str(error),status='pending')
        try:
            checkpoint()
            newer=capture();checkpoint()
            revised=select(newer['rgb'],record,directory=directory,checkpoint=checkpoint,
                review_context=dict(reason=str(error),request='Select full-width visible region of the SAME spine with textured interior; keep both spine edges, no cover or text-only strip'),
                reference_rgb=frame['rgb'],reference_mask=chosen['mask'])
            checkpoint()
            if revised.get('identity_confirmed') is not True:
                raise ServoError('GPT initialization did not confirm the same book')
            candidate=make_lock(newer['rgb'],revised,factory=factory)
            fresh=capture();checkpoint()
            if fresh['captured_at']<=newer['captured_at']:
                raise ServoError('GPT initialization requires a newer verification image')
            try:observation=candidate.observe(fresh['rgb'])
            except (FeaturePointsExhausted,CenterlineGeometryError) as failure:
                raise ServoError(f'GPT initialization verification failed: {failure}') from failure
            checkpoint()
            if max_frame_age_seconds is not None and clock()-fresh['captured_at']>max_frame_age_seconds:
                raise ServoError('GPT initialization verification image expired')
            candidate.selection_override=revised
            cv2.imwrite(str(directory/'reference.png'),cv2.cvtColor(newer['rgb'],cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(directory/'selected_mask.png'),np.uint8(revised['mask'])*255)
            record.update(status='recovered',inliers=int(observation['inliers']))
            return candidate
        except BaseException as failure:
            record.update(status='failed',error=str(failure));raise
        finally:
            (directory/'initialization.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


def wait_stationary(read,checkpoint,record,deadline,reference_z,z_limit,*,clock=time.monotonic,sleep=time.sleep):
    """Observe a held arm using distinct feedback samples before exposing it.

    Arrival within a target ball does not imply zero velocity. This check sends
    no commands and does not relax the subsequent 1mm / 0.5deg image guard.
    """
    stop=min(deadline,clock()+3.);window=[];last_stamp=-np.inf
    while True:
        checkpoint();state=read();now=clock()
        if now>=stop:
            raise ServoError('GPT stationary hold timed out; no fresh stable 0.3s window before capture')
        stamp=float(state['pose_sample_monotonic'])
        if not np.isfinite(stamp) or not 0<=now-stamp<=.25:
            raise ServoError('GPT settling requires fresh measured feedback')
        q=np.asarray(state['q'],float);angles=np.asarray(state['angles'],float)
        if abs(q[2]-reference_z)>z_limit:
            raise ServoError('Height drift while waiting for GPT stationary hold')
        if stamp>last_stamp:
            last_stamp=stamp
            rotation=Rotation.from_euler('xyz',angles,degrees=True)
            # Bound each pair inside the accepted window, including rotation.
            translation=max((float(np.linalg.norm(q-v[1])) for v in window),default=0.)
            angle=max((float(np.degrees((rotation*v[2].inv()).magnitude())) for v in window),default=0.)
            if translation>.25 or angle>.15:window=[]
            window.append((stamp,q.copy(),rotation))
            duration=stamp-window[0][0]
            record['stationary_hold']=dict(samples=len(window),stable_seconds=duration,
                position_span_mm=translation,orientation_span_deg=angle,
                required_seconds=.3,position_span_limit_mm=.25,orientation_span_limit_deg=.15)
            if len(window)>=3 and duration>=.3:
                return state
        sleep(.01)


def verify_recovery_frames(candidate, reference_stamp, transport, capture, checkpoint,
                           record, deadline, retry_seconds, *, clock, sleep):
    """Retry only image expiry while holding the SAME GPT selection and arm pose.

    The board entry opts in. Feedback/motion/semantic faults propagate; image
    age and the grasp deadline are unchanged. No targets or GPT calls here.
    """
    from cart_book_grasp_v3_image_freshness import ImageFreshnessRecovery
    if not np.isfinite(retry_seconds) or retry_seconds <= 0:
        raise ServoError('Invalid GPT verification retry budget')
    publish=getattr(transport,'log',lambda event:None)
    def event(value):
        now=clock()
        value=dict(value,event='gpt_verification_'+value['event'],checked_at=now)
        if attempts and 'frame_stamp' in attempts[-1]:
            last=attempts[-1]
            value.update(attempt=last['attempt'],frame_stamp=last['frame_stamp'],
                         source_age_seconds=now-last['frame_stamp'])
            if 'tracking_finished_at' in last:
                value['post_tracking_seconds']=now-last['tracking_finished_at']
        record.setdefault('verification_events',[]).append(value)
        publish(value)
    settings=dict(frame_max_age_seconds=transport.s['frame_max_age_seconds'],
                  pair_recovery_timeout_seconds=retry_seconds)
    guard=ImageFreshnessRecovery(settings,clock,sleep,checkpoint,event,deadline)
    previous=reference_stamp
    attempts=record.setdefault('verification_attempts',[])
    def observe():
        nonlocal previous
        timing=dict(attempt=len(attempts)+1,started_at=clock(),
                    frame_max_age_seconds=settings['frame_max_age_seconds'])
        try:
            data=capture(guard.check)
            timing['capture_returned_at']=clock()
            stamp=float(data[0]['captured_at'])
            if not np.isfinite(stamp):
                timing['invalid_frame_stamp']=repr(stamp)
                guard.validate(stamp,previous,stage='gpt_verification_capture')
            timing.update(frame_stamp=stamp,
                          source_age_at_pair_return_seconds=clock()-stamp,
                          capture_timing=data[0].get('capture_timing',{}))
            guard.validate(stamp,previous,stage='gpt_verification_capture')
            previous=stamp
            timing['tracking_started_at']=clock()
            observation=candidate.observe(data[0]['rgb'])
            timing['tracking_finished_at']=clock()
            # The outer guard checks age AGAIN after this callback, including
            # its final feedback check and diagnostic handoff.
            return dict(stamp=stamp,data=data,observation=observation)
        except BaseException as error:
            timing['error']=f'{type(error).__name__}: {error}'
            raise
        finally:
            timing['attempt_finished_at']=clock()
            if 'frame_stamp' in timing:
                timing['source_age_at_attempt_end_seconds']=clock()-timing['frame_stamp']
            attempts.append(timing)
            publish(dict(timing,event='gpt_verification_frame'))
    result=guard.observe(observe,reference_stamp)
    record['verification_age_seconds']=clock()-result['stamp']
    publish(dict(event='gpt_verification_accepted',attempts=len(attempts),
                 frame_stamp=result['stamp'],source_age_seconds=record['verification_age_seconds'],
                 post_tracking_seconds=clock()-attempts[-1]['tracking_finished_at'],
                 frame_max_age_seconds=settings['frame_max_age_seconds']))
    return result['data'],result['observation']


def recover_tracking(old_lock, transport, capture, select, checkpoint, directory,
                     deadline, *, reference_z, clock=time.monotonic, factory=FeatureCenterlineLock,
                     sleep=time.sleep, verification_retry_seconds=0.):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    record={'status':'pending','trigger':'tracking_geometry_failure'};anchor=None
    def checked():
        checkpoint()
        if clock()>=deadline:raise ServoError('GPT tracking recovery exceeded total time budget')
        state=transport.read()
        if anchor is not None:
            angle=np.degrees((Rotation.from_euler('xyz',state['angles'],degrees=True)*
                              Rotation.from_euler('xyz',anchor['angles'],degrees=True).inv()).magnitude())
            delta=np.asarray(state['q'])-anchor['q'];distance=float(np.linalg.norm(delta))
            record['motion_guard']=dict(anchor_xyz_mm=anchor['q'].tolist(),actual_xyz_mm=np.asarray(state['q']).tolist(),
                delta_xyz_mm=delta.tolist(),position_drift_mm=distance,orientation_drift_deg=float(angle),
                position_limit_mm=1.,orientation_limit_deg=.5,
                pose_sample_monotonic=state.get('pose_sample_monotonic'))
            if distance>1. or angle>.5:
                raise ServoError(f'Arm moved during GPT tracking recovery; image region invalid: '
                                 f'drift={distance:.3f}mm (limit=1.000mm), rotation={angle:.3f}deg (limit=0.500deg)')
        return state
    try:
        state=checked()
        # A confirmed finite hold replaces the preceding rolling target without
        # latching the terminal abort state. The adapter maintains this hold.
        goal=state['q'].copy();goal[2]=reference_z
        if abs(state['q'][2]-reference_z)>transport.s['max_z_drift_mm']:
            raise ServoError('Height drift before GPT recovery hold')
        if not transport.finish(goal,state['roll']):
            raise ServoError('GPT tracking recovery hold not confirmed')
        settled=wait_stationary(transport.read,checkpoint,record,deadline,reference_z,
            transport.s['max_z_drift_mm'],clock=clock,sleep=sleep)
        print(f"GPT 恢复：已确认连续静止 {record['stationary_hold']['stable_seconds']:.3f}s，再采集识别图像。",flush=True)
        anchor=dict(q=np.asarray(settled['q'],float).copy(),
                    angles=np.asarray(settled['angles'],float).copy())
        review=None
        for attempt in range(2):
            folder=directory/f'attempt_{attempt+1:02d}';folder.mkdir(parents=True,exist_ok=True)
            frame,*_=capture(checked);checked()
            cv2.imwrite(str(folder/'gpt_reference.png'),cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2BGR))
            kwargs={}
            if review is not None:
                kwargs=dict(review_context=review,reference_rgb=getattr(old_lock,'reference_rgb',None),
                            reference_mask=getattr(old_lock,'reference_mask',None))
            chosen=select(frame['rgb'],record,directory=folder,checkpoint=checked,**kwargs)
            checked()
            try:
                candidate=make_lock(frame['rgb'],chosen,factory=factory)
                jump=abs(candidate.last_u-old_lock.last_u)>80 or abs(candidate.last_angle-old_lock.last_angle)>12
                if jump and not (review is not None and chosen.get('identity_confirmed') is True):
                    raise CenterlineGeometryError('GPT recovery centerline disagrees with previously tracked book')
                # Never use an API-aged frame to update a moving target.
                if verification_retry_seconds > 0:
                    fresh,observation=verify_recovery_frames(candidate,frame['captured_at'],
                        transport,capture,checked,record,deadline,verification_retry_seconds,
                        clock=clock,sleep=sleep)
                else:
                    fresh=capture(checked);checked()
                    if fresh[0]['captured_at']<=frame['captured_at']:
                        raise ServoError('GPT recovery requires a newer verification image')
                    observation=candidate.observe(fresh[0]['rgb'])
                    checked()
                    if clock()-fresh[0]['captured_at']>transport.s['frame_max_age_seconds']:
                        raise ServoError('GPT recovery verification image expired')
            except (InitialFeaturesUnavailable,FeaturePointsExhausted,CenterlineGeometryError) as failure:
                if attempt or getattr(old_lock,'reference_rgb',None) is None:
                    raise TrackingRecoveryRejected(f'GPT recovery verification failed: {failure}') from failure
                review=dict(reason=str(failure),previous_center_u=float(old_lock.last_u),
                            previous_angle_deg=float(old_lock.last_angle),request='Compare reference and current book; repair the same spine region')
                record.setdefault('retries',[]).append(review)
                continue
            cv2.imwrite(str(folder/'selected_mask.png'),np.uint8(chosen['mask'])*255)
            cv2.imwrite(str(folder/'verified_wrist.png'),cv2.cvtColor(fresh[0]['rgb'],cv2.COLOR_RGB2BGR))
            record.update(status='recovered',reference_stamp=float(frame['captured_at']),
                          verification_stamp=float(fresh[0]['captured_at']),inliers=int(observation['inliers']))
            return candidate,fresh,observation
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        (directory/'recovery.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
