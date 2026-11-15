"""Opt-in manual-cell pose recovery on155; no corrective motion or rebasing."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from cart_book_grasp_v3_servo import ServoError
from cart_book_grasp_v3_centerline import roll_and_residual
from cart_book_grasp_v3_image_freshness import PoseRecoveryRetry, StaleImage

# These bound a hold/recheck, not the permitted pose for sending targets.
SETTINGS = dict(hard_residual_deg=1.2, timeout_seconds=2., stable_seconds=.3,
                stable_samples=5, velocity_rad_s=.02, position_span_mm=.25,
                orientation_span_deg=.15)


class AxisRecovery:
    def __init__(self, settings, normal_limit, image_recovery, checkpoint, sleep, log):
        # A versioned policy: callers cannot silently widen it in a worker job.
        if settings != SETTINGS or normal_limit != 1.:
            raise ValueError('Unsupported axis recovery policy/normal tolerance')
        self.settings = dict(settings)
        self.normal_limit = normal_limit
        self.images, self.checkpoint, self.sleep, self.log = image_recovery, checkpoint, sleep, log
        self.clock = image_recovery.clock
        self.resume_after = -np.inf
        self.pending_visual_reset = False
        self.recovery_path_mm = 0.

    def validate_frame(self, stamp):
        if stamp <= self.resume_after:
            raise StaleImage('Image predates confirmed axis hold; recapture')

    def recover(self, transport, initial, error):
        cfg = self.settings
        evidence = dict(residual_deg=error.residual_deg, normal_limit_deg=self.normal_limit,
                        hard_limit_deg=cfg['hard_residual_deg'],
                        base_deg=np.asarray(transport.base).tolist(),
                        actual_deg=initial['angles'].tolist())
        if error.residual_deg >= cfg['hard_residual_deg']:
            self.log(dict(event='axis_recovery_stop', reason='hard_limit', **evidence))
            raise ServoError('Tool-axis residual hard limit exceeded: '+str(error)) from error
        began = self.clock()
        deadline = min(began+cfg['timeout_seconds'], self.images.total_deadline)
        self.log(dict(event='axis_recovery_start', timeout_seconds=cfg['timeout_seconds'],
                      policy='pause_targets_ack_hold_settle_recheck', **evidence))
        latest = dict(evidence)
        try:
            with self.images.held_tracking_recovery(event='image_recovery_axis_hold_time'):
                self.checkpoint()
                hold = getattr(transport.robot, 'hold_for_axis_recovery', None)
                if hold is None or hold() is not True:
                    raise ServoError('Axis recovery hold not acknowledged')
                acknowledged = self.clock()
                transport.pending = None
                transport.frame_stamp = None
                self.log(dict(event='axis_recovery_hold_acknowledged', elapsed_seconds=acknowledged-began))
                window = deque()
                last_source = initial["pose_sample_monotonic"]
                last_stamp = acknowledged
                previous = initial['q']
                held_reported = False
                while True:
                    self.checkpoint() # Maintains ownership; all backend faults propagate.
                    if self.clock() >= deadline:
                        raise ServoError('Tool-axis hold/recheck timed out; normal pose not confirmed')
                    state = transport._read_feedback() # Original status, RT age and provenance checks.
                    _, residual = roll_and_residual(transport.base, state['angles'])
                    velocity = np.asarray(transport.last_state.get('joint_velocity_rad_s'), float)
                    if velocity.size != 14 or not np.isfinite(velocity).all():
                        raise ServoError('Axis recovery requires valid measured joint velocities')
                    speed = float(np.max(np.abs(velocity)))
                    latest = dict(residual_deg=residual, max_joint_velocity_rad_s=speed,
                                  feedback_age_seconds=self.clock()-state['pose_sample_monotonic'])
                    if residual >= cfg['hard_residual_deg']:
                        raise ServoError(f'Tool-axis residual hard limit during hold: {residual:.6f}deg')
                    if np.linalg.norm(state['q']-initial['q'])>transport.c['max_pose_error_mm']:
                        raise ServoError('Axis recovery hold position drift exceeded')
                    angle=float(np.degrees((Rotation.from_euler('xyz',initial['angles'],degrees=True).inv()*
                                            Rotation.from_euler('xyz',state['angles'],degrees=True)).magnitude()))
                    if angle>transport.s['lookahead_roll_deg']+self.normal_limit:
                        raise ServoError('Axis recovery hold orientation drift exceeded')
                    stamp = state['pose_sample_monotonic']
                    if stamp < last_source:raise ServoError('Axis recovery feedback timestamp moved backwards')
                    last_source = stamp
                    if stamp > last_stamp:
                        self.recovery_path_mm += float(np.linalg.norm(state['q']-previous))
                        previous = state['q'].copy()
                        last_stamp = stamp
                        if speed > cfg['velocity_rad_s']:
                            window.clear()
                        else:
                            window.append((stamp,state,residual))
                            while len(window)>1 and stamp-window[1][0]>=cfg['stable_seconds']:
                                window.popleft()
                        position_span = 0.; orientation_span = 0.
                        if window:
                            positions = np.array([v[1]['q'] for v in window])
                            position_span = float(np.linalg.norm(np.ptp(positions,axis=0)))
                            rotations = Rotation.from_euler('xyz',[v[1]['angles'] for v in window],degrees=True)
                            # Pairwise bound, rather than undercounting opposite sides of the window.
                            orientation_span = max(float(np.degrees((r.inv()*rotations).magnitude()).max()) for r in rotations)
                        stable = (len(window)>=cfg['stable_samples'] and
                                  stamp-window[0][0]>=cfg['stable_seconds'] and
                                  position_span<=cfg['position_span_mm'] and
                                  orientation_span<=cfg['orientation_span_deg'])
                        latest.update(stable_samples=len(window), position_span_mm=position_span,
                                      orientation_span_deg=orientation_span)
                        if stable and not held_reported:
                            self.log(dict(event='axis_recovery_hold_confirmed', **latest))
                            held_reported = True
                        # Every sample of the settled window must satisfy the original 1 degree limit.
                        if stable and all(v[2]<=self.normal_limit for v in window):
                            self.resume_after = self.clock()
                            self.pending_visual_reset = True
                            self.images.axis_hold_pending = True
                            transport.timeline.samples.clear()
                            self.log(dict(event='axis_recovery_complete', elapsed_seconds=self.clock()-began,
                                          resume_after=self.resume_after, **latest))
                            break
                    self.sleep(.01)
        except BaseException as failure:
            self.log(dict(event='axis_recovery_stop', reason=str(failure),
                          elapsed_seconds=self.clock()-began, **latest))
            raise
        # read(), ready(), send() and finish() must ALL discard their old work.
        raise PoseRecoveryRetry('Axis hold confirmed; discard old target and reacquire image')

    def consume_visual_reset(self):
        result = self.pending_visual_reset
        self.pending_visual_reset = False
        self.images.axis_hold_pending = False
        return result
