"""Bounded local evidence for the cockpit state/heartbeat path; no control I/O."""
from collections import deque
import time

HEARTBEAT_PERIOD_S=.05
HEARTBEAT_STATE_BUDGET_S=.15
HEARTBEAT_SEND_TIMEOUT_S=.2
STATE_TIMEOUT_S=1.


class ControlTiming:
    def __init__(self):
        self.events=deque(maxlen=120)
        self.source_tick=None
        self.received=None
        self.best_clock_offset=None
        self.states=0
        self.sent=0
        self.skipped=0

    def observe(self,state,now):
        tick=state.get('challenge')
        if type(tick) is not int:
            raise RuntimeError('驾驶舱状态缺少有效时效标记')
        if self.source_tick is not None and tick<self.source_tick:
            raise RuntimeError('驾驶舱状态时序倒退；停止本会话')
        offset=now-tick*.05
        self.best_clock_offset=offset if self.best_clock_offset is None else min(self.best_clock_offset,offset)
        self.events.append(dict(event='state',local_monotonic_s=now,
            read_gap_ms=None if self.received is None else (now-self.received)*1000,
            source_advance_ms=None if self.source_tick is None else (tick-self.source_tick)*50,
            extra_delay_ms=max(0.,offset-self.best_clock_offset)*1000,
            rt_feedback_age_ms=state.get('age_ms'),fresh=state.get('fresh'),
            active=state.get('active'),mine=state.get('mine'),
            planner_enabled=state.get('planner',{}).get('enabled')))
        self.received=now;self.source_tick=tick;self.states+=1

    def extra_age(self,now):
        if self.source_tick is None:return float('inf')
        # Relative to the fastest previously seen state. This cancels the two
        # hosts' monotonic clock offset; it is NOT an absolute one-way latency.
        return max(0.,now-self.source_tick*.05-self.best_clock_offset)

    def heartbeat(self,event,now,**details):
        self.events.append(dict(event=event,local_monotonic_s=now,
            state_age_ms=(now-self.received)*1000,extra_delay_ms=self.extra_age(now)*1000,
            **details))
        if event=='heartbeat_sent':self.sent+=1
        if event=='heartbeat_skipped_old_state':self.skipped+=1

    def snapshot(self,error,state):
        recent=(state or {}).get('stop_diagnostics',{}).get('recent',[])
        stop=recent[-1] if recent else {}
        now=time.monotonic()
        return dict(unix_s=time.time(),event='launcher_control_failure',error=str(error),
            state_read_age_ms=None if self.received is None else (now-self.received)*1000,
            source_extra_age_ms=None if self.source_tick is None else self.extra_age(now)*1000,
            states_received=self.states,heartbeats_sent=self.sent,old_states_skipped=self.skipped,
            note='extra_delay is relative to fastest prior state; read time is not kernel arrival time',
            last_backend_stop={k:stop.get(k) for k in
                               ('unix_s','reason','source','rt_feedback_age_ms')},
            planner_fault=stop.get('planner_fault'),
            events=list(self.events))
