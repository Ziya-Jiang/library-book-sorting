"""Stop evidence captured before reset; bounded async persistence, never a control lease."""
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid


def capture(controller, reason, source):
    now=time.monotonic();wall=time.time()
    bridge=getattr(controller,'planner',None)
    state=getattr(controller,'state',None) or {}
    received=getattr(controller,'received',0.)
    session=getattr(bridge,'session','')
    entry = dict(event='control_stop',id=uuid.uuid4().hex,unix_s=wall,
        utc=datetime.fromtimestamp(wall,timezone.utc).isoformat(),time_monotonic=now,
        source=source,reason=str(reason),epoch=getattr(controller,'epoch',None),
        session_tag=hashlib.sha256(str(session).encode()).hexdigest()[:12] if session else None,
        active=bool(getattr(controller,'active',False)),planner_enabled=bool(getattr(bridge,'enabled',False)),
        rt_feedback_age_ms=(now-received)*1000 if received else None,
        rt_timestamp=state.get('timestamp'),rt_system=state.get('system'),rt_diagnostics=state.get('diagnostics'),
        target_id=getattr(getattr(bridge,'engine',None),'last_target_id',None))
    # Capture before disarm replaces the gate. Do not attach an older fault
    # to an unrelated manual stop or a normal visual ownership release.
    fault = getattr(bridge, 'last_fault', None)
    if isinstance(fault, dict) and fault.get('reason') == str(reason):
        entry['planner_fault'] = copy.deepcopy(fault)
    return entry


def rejection(controller, message, error):
    """Return the matching stop cause, not an unrelated previous session's stop."""
    result = dict(type='error', message=str(error))
    detail = getattr(controller, '_stop_context', None)
    if detail is None:
        history = getattr(getattr(controller, 'stop_journal', None), 'history', ())
        epoch = message.get('epoch') if isinstance(message, dict) else None
        detail = next((row for row in reversed(history) if row.get('epoch') == epoch), None)
    if (isinstance(message, dict) and isinstance(detail, dict)
            and type(message.get('epoch')) is int
            and message['epoch'] != getattr(controller, 'epoch', None)
            and message['epoch'] == detail.get('epoch')):
        result['stop_detail'] = copy.deepcopy(detail)
    return result


class StopJournal:
    def __init__(self,root):
        self.path=Path(root)/'runtime'/'control_stops.jsonl'
        self.history=deque(maxlen=32);self.pending=queue.Queue(maxsize=128)
        self.dropped=0;self.write_error=None;self.persisted_id=None
        self.closing=threading.Event()
        self.thread=threading.Thread(target=self._writer,name='stop-diagnostics',daemon=True)
        self.thread.start()

    def record(self,entry):
        self.history.append(entry)
        try:self.pending.put_nowait(entry)
        except queue.Full:self.dropped+=1

    def snapshot(self):
        return dict(recent=list(self.history),persisted_id=self.persisted_id,
                    dropped=self.dropped,write_error=self.write_error,path=str(self.path))

    def _writer(self):
        while not self.closing.is_set() or not self.pending.empty():
            try:entry=self.pending.get(timeout=.05)
            except queue.Empty:continue
            try:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                self.persisted_id=entry['id']
            except Exception as error:self.write_error=f'{type(error).__name__}: {error}'
            finally:self.pending.task_done()

    def close(self):
        self.closing.set();self.thread.join(timeout=2.)
