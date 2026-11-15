"""Drop only expired heartbeats; never renew a lease from rejected traffic."""
from collections import deque
from pathlib import Path
import math
import queue
import threading
import time
import uuid
from .stop_diagnostics import StopJournal

CHALLENGE_WINDOW_S=.3


class HeartbeatJournal(StopJournal):
    def __init__(self,root):
        self.path=Path(root)/'runtime'/'heartbeat_discards.jsonl'
        self.history=deque(maxlen=1);self.pending=queue.Queue(maxsize=128)
        self.discarded_count=0
        self.dropped=0;self.write_error=None;self.persisted_id=None
        self.closing=threading.Event()
        self.thread=threading.Thread(target=self._writer,name='heartbeat-diagnostics',daemon=True)
        self.thread.start()

    def record(self,entry):
        self.discarded_count+=1
        super().record(entry)

    def snapshot(self):
        # Periodic state carries only the latest discard, not a growing batch.
        return dict(super().snapshot(),discarded_count=self.discarded_count)


def expired_heartbeat(controller,owner,message,*,preparation=False,received_monotonic=None):
    """Return a nonfatal diagnostic only for an expired, otherwise owned request.

    Caller holds the controller lock. This never changes heartbeat/deadline,
    command receipt, owner, active flag, epoch, or targets.
    """
    action='prepare_heartbeat' if preparation else 'heartbeat'
    if message.get('action')!=action:raise ValueError('Not a heartbeat request')
    if message.get('epoch')!=controller.epoch:
        raise ValueError('停止后请重新接管控制')
    if preparation:
        if not controller._reserved() or controller._prepare_owner!=owner:
            raise ValueError('启动准备控制权已失效；取消本次启动，不自动重新接管')
    elif controller.owner!=owner or not controller.active:
        raise ValueError('请先接管关节控制')
    challenge=message.get('challenge')
    if type(challenge) is not int:raise ValueError('控制请求时效标记无效')
    try:issued=challenge*.05
    except OverflowError:raise ValueError('控制请求时效标记无效') from None
    now=time.monotonic();age=now-issued
    if not math.isfinite(age) or age<0:raise ValueError('控制请求时效标记无效')
    if age<=CHALLENGE_WINDOW_S:return None
    received=now if received_monotonic is None else received_monotonic
    entry=dict(type='heartbeat_discarded',event='expired_heartbeat_discarded',id=uuid.uuid4().hex,
        action=action,epoch=controller.epoch,unix_s=time.time(),challenge=challenge,
        challenge_slot_monotonic_s=issued,received_monotonic_s=received,processed_monotonic_s=now,
        challenge_age_at_receive_ms=(received-issued)*1000,challenge_age_at_process_ms=age*1000,
        receive_to_process_ms=max(0.,now-received)*1000,challenge_limit_ms=CHALLENGE_WINDOW_S*1000,
        rt_feedback_age_ms=(now-controller.received)*1000 if controller.received else None,
        lease_renewed=False,reason='expired_challenge',
        note='receive timestamp is application WebSocket dispatch, not kernel arrival; challenge is a 50ms slot')
    journal=getattr(controller,'heartbeat_journal',None)
    if journal is not None:journal.record(entry)
    return entry


def action_response(message,result):
    if isinstance(result,dict) and result.get('type')=='heartbeat_discarded':return result
    if message.get('action')!='heartbeat':return dict(type='ok',id=message.get('id'))
    return None
