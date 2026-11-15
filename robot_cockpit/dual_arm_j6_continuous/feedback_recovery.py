"""Continuous-J6 feedback gate. Never accepts an old sample as current feedback."""
import math

FRESH_SECONDS = .25


class FeedbackFault(RuntimeError):
    def __init__(self, reason, detail):
        self.feedback_detail = dict(detail, reason=reason)
        super().__init__('连续J6反馈保护：'+reason+'；来源='+','.join(detail.get('sources', [])))


def feedback_ages(sample, arm, now):
    ages = {}
    def age(name, stamp):
        if stamp is None:
            ages[name] = None
        elif isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise FeedbackFault('invalid_timestamp', dict(sources=[name], ages_ms=ages))
        else:
            value = now-stamp
            if value < 0:
                raise FeedbackFault('future_timestamp', dict(sources=[name], ages_ms=ages))
            ages[name] = value*1000
    age('udp_receive', (sample or {}).get('received_monotonic'))
    age('pose_sample', (sample or {}).get('pose_sample_monotonic'))
    age('websocket_receive', arm.stamp)
    ages['websocket_source_extra'] = arm.timing.extra_age(now)*1000
    rt_age = (arm.state or {}).get('age_ms')
    ages['rt_feedback'] = (rt_age+ages['websocket_receive']
                           if isinstance(rt_age, (int, float)) and not isinstance(rt_age, bool)
                           and ages['websocket_receive'] is not None else None)
    for name,value in ages.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise FeedbackFault('invalid_age', dict(sources=[name], ages_ms=ages))
    fk_stamp=(sample or {}).get('pose_fk_completed_monotonic')
    fk_age=((now-fk_stamp)*1000 if isinstance(fk_stamp,(int,float)) and math.isfinite(fk_stamp) else None)
    return dict(ages_ms=ages,
                sources=[k for k,v in ages.items() if v is None or v>FRESH_SECONDS*1000],
                fresh_limit_ms=FRESH_SECONDS*1000,
                received_monotonic=(sample or {}).get('received_monotonic'),
                pose_sample_monotonic=(sample or {}).get('pose_sample_monotonic'),
                websocket_received_monotonic=arm.stamp,
                pose_fk_completed_monotonic=fk_stamp, pose_fk_completed_age_ms=fk_age,
                epoch=(arm.state or {}).get('epoch'),
                rt_timestamp=((arm.state or {}).get('state') or {}).get('timestamp'))


class FeedbackGate:
    def __init__(self, make_window, record, loss_seconds=1., settle_seconds=8.):
        self.make_window, self.record = make_window, record
        self.loss_seconds, self.settle_seconds = loss_seconds, settle_seconds
        self.ready = True
        self.generation = 0
        self.began = self.fresh_began = self.gap_began = None
        self.window = None
        self.last_pose = None
        self.latest = None
        self.last_sources = None

    def update(self, detail, sample, state, now):
        self.latest = detail
        sources = detail['sources']
        if sources:
            if self.gap_began is None:self.gap_began=now
            if self.began is None:
                self.began = now
                self.record('feedback_pause', **detail, policy='no_new_targets_wait_fresh_then_settle')
            if self.window is not None or self.ready:
                self.generation += 1
            self.ready = False
            self.window = None
            self.last_pose = None
            self.fresh_began = None
            if self.last_sources != sources:
                self.record('feedback_sources', **detail)
                self.last_sources = list(sources)
            expired = [k for k,v in detail['ages_ms'].items()
                       if v is not None and v>self.loss_seconds*1000]
            if expired or now-self.gap_began>self.loss_seconds:
                raise FeedbackFault('persistent_feedback_loss', dict(detail, expired_sources=expired,
                                    loss_limit_s=self.loss_seconds, elapsed_s=now-self.gap_began))
            if now-self.began>self.settle_seconds:
                raise FeedbackFault('feedback_never_continuously_settled', dict(detail,settle_limit_s=self.settle_seconds))
            return False
        self.gap_began=None
        if self.began is None:
            self.ready = True
            return True
        if self.window is None:
            self.window = self.make_window()
            self.fresh_began = now
            self.record('feedback_fresh_wait_settled', **detail)
        # A frozen UDP/FK sample cannot establish a continuously settled window.
        stamps=(sample['received_monotonic'],sample['pose_sample_monotonic'])
        if self.last_pose is None or (stamps[0]>self.last_pose[0] and stamps[1]>self.last_pose[1]):
            self.last_pose = stamps
            if self.window.observe(state, now):
                self.record('feedback_recovered', **detail, elapsed_s=now-self.began,
                            settled_feedback=self.window.latest)
                self.ready = True
                self.began = self.fresh_began = None
                self.window = None
                self.last_sources = None
                return True
        if now-self.began>self.settle_seconds:
            raise FeedbackFault('feedback_recovered_but_not_settled',
                                dict(detail, settle_limit_s=self.settle_seconds))
        return False
