"""Short, non-motion startup reservation. STOP always wins; no automatic rearm."""
import time
from .heartbeat_policy import expired_heartbeat

PREPARATION_LEASE_S = 1.


def with_reservation(base):
    class ReservedController(base):
        def _reserved(self):
            return (getattr(self, '_prepare_owner', None) is not None
                    and getattr(self, '_prepare_epoch', None) == self.epoch
                    and time.monotonic() < self._prepare_deadline)

        def stop(self, *args, **kwargs):
            self._prepare_owner = None
            return super().stop(*args, **kwargs)

        def action(self, owner, message, *, received_monotonic=None):
            with self.lock:
                action = message.get('action')
                if action in ('stop', 'stop_rt'):
                    return super().action(owner, message, received_monotonic=received_monotonic)
                if action == 'prepare_start':
                    self.verify(owner, message, require_owner=False)
                    if self.active or self.planner.enabled or self._reserved():
                        raise ValueError('已有控制或启动准备会话；本次不抢占')
                    self._prepare_owner = owner
                    self._prepare_epoch = self.epoch
                    self._prepare_deadline = time.monotonic()+PREPARATION_LEASE_S
                    return
                if action in ('prepare_heartbeat', 'prepare_cancel', 'prepared_planner_enable'):
                    if action=='prepare_heartbeat':
                        discarded=expired_heartbeat(self,owner,message,preparation=True,received_monotonic=received_monotonic)
                        if discarded is not None:return discarded
                    else:
                        self.verify(owner, message, require_owner=False)
                    if not self._reserved() or self._prepare_owner != owner:
                        raise ValueError('启动准备控制权已失效；取消本次启动，不自动重新接管')
                    if action == 'prepare_cancel':
                        self._prepare_owner = None
                        return
                    if action == 'prepare_heartbeat':
                        self._prepare_deadline = time.monotonic()+PREPARATION_LEASE_S
                        return
                    # Delegate all original feedback, stationary and joint checks.
                    result = super().action(owner, dict(message, action='planner_enable'), received_monotonic=received_monotonic)
                    self._prepare_owner = None
                    return result
                if self._reserved():
                    raise ValueError('一键抓取正在准备，由启动器自动使能；如需取消请先停止')
                return super().action(owner, message, received_monotonic=received_monotonic)

        def snapshot(self, owner=None):
            with self.lock:
                result = super().snapshot(owner)
                reserved = self._reserved()
                result['preparation'] = dict(active=reserved,
                    mine=reserved and self._prepare_owner == owner,
                    remaining_ms=max(0., (getattr(self, '_prepare_deadline', 0.)-time.monotonic())*1000)
                        if reserved else 0.)
                return result
    return ReservedController
