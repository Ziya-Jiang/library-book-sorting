"""Bounded stale-image recovery; no robot commands or device access."""
import math
from contextlib import contextmanager
from cart_book_grasp_v3_servo import ServoError


class ImageRecoveryTimedOut(ServoError):
    """Only image retry expiry, distinct from feedback and total-time expiry."""


class StaleImage(ServoError):
    """A valid timestamp is too old or has already been consumed."""


class ImageMotionExpired(StaleImage):
    """A paired image is too far behind current motion to predict safely."""
    def __init__(self, message, *, tracking_recovered=False):
        super().__init__(message)
        self.tracking_recovered = tracking_recovered


class PoseRecoveryRetry(StaleImage):
    """A confirmed pose hold invalidated the image and any computed target."""
    tracking_recovered = True


class ImageFreshnessRecovery:
    def __init__(self, settings, clock, sleep, checkpoint, log, total_deadline):
        self.max_age = settings['frame_max_age_seconds']
        self.timeout = settings['pair_recovery_timeout_seconds']
        self.clock, self.sleep = clock, sleep
        self.checkpoint, self.log = checkpoint, log
        self.total_deadline = total_deadline
        self.started = None
        self.attempts = 0
        self.axis_hold_pending = False

    def check(self):
        self.checkpoint()
        now = self.clock()
        if now >= self.total_deadline:
            raise ServoError('V3.2 total time budget exceeded')
        if self.started is not None and now - self.started >= self.timeout:
            self.log(dict(event='image_recovery_timeout', attempts=self.attempts,
                          elapsed_seconds=now-self.started))
            if self.axis_hold_pending:
                raise ServoError('Axis recovery fresh-image verification timed out; forward cancelled')
            raise ImageRecoveryTimedOut('V3.2 image recovery timed out; no new motion targets sent during recovery')

    @contextmanager
    def held_tracking_recovery(self, event='image_recovery_semantic_time'):
        """Exclude guarded semantic recovery, not ordinary frame retries.

        Preserve remaining retry time and the global grasp deadline. The caller
        must hold, verify feedback, and obtain a fresh frame before returning.
        """
        self.check()
        began = self.clock()
        try:
            yield
        finally:
            duration = self.clock() - began
            if self.started is not None:
                self.started += duration
            self.log(dict(event=event, excluded_seconds=duration,
                          policy='held_tracking_recovery_separate_budget'))
        # Preserve underlying failures rather than masking them in finally.
        self.check()

    def validate(self, stamp, previous=float('-inf'), stage='control'):
        self.check()
        now = self.clock()
        if not math.isfinite(stamp) or stamp <= 0 or stamp > now:
            raise ServoError('V3.2 invalid image timestamp')
        age = now-stamp
        if stamp <= previous or age > self.max_age:
            raise StaleImage(f'V3.2 stale image at {stage}: age={age:.6f}s, '
                             f'limit={self.max_age:.6f}s, duplicate={stamp <= previous}')

    def observe(self, callback, previous):
        tracking_recovered = False
        while True:
            self.check()
            try:
                result = callback()
                tracking_recovered |= bool(result.get('tracking_recovered', False))
                self.validate(float(result['stamp']), previous)
            except StaleImage as error:
                tracking_recovered |= bool(getattr(error, 'tracking_recovered', False))
                self.discard(error)
                self.sleep(.005)
                continue
            except Exception:
                # A late point-loss result cannot escape the recovery deadline.
                self.check()
                raise
            if self.started is not None:
                self.log(dict(event='image_recovered', attempts=self.attempts,
                              elapsed_seconds=self.clock()-self.started,
                              frame_stamp=float(result['stamp'])))
            self.started = None
            self.attempts = 0
            if tracking_recovered:
                # GPT may have replaced the tracker before snapshot I/O aged
                # its first result. Do not lose the Jacobian reset notification.
                result['tracking_recovered'] = True
            return result

    def discard(self, error):
        self.check()
        if self.started is None:
            self.started = self.clock()
            self.log(dict(event='image_recovery_start', timeout_seconds=self.timeout,
                          policy='pause_target_updates_recapture'))
        self.attempts += 1
        self.log(dict(event='image_discarded', reason=str(error), attempts=self.attempts))
