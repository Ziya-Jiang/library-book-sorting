"""UDP transport subprocess; private inherited IPC only, no CUDA/solver."""
import argparse
import copy
from collections import deque
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import sys
import threading
import time

# Arm before importing transport dependencies or opening any UDP socket.
if __name__ == '__main__':
    from planner_lifecycle import arm_parent_death
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--parent-pid', type=int, required=True)
    parent_args, _ = early.parse_known_args()
    arm_parent_death(parent_args.parent_pid)

from v3_1_panel_process import (read_message, write_message, REQUEST_AGE,
                               publisher_fingerprint, process_identity)
from v3_1_board_client import BoardPanel, BoardTransport


class LatestDiagnostic:
    """One pending snapshot; disk latency must never block the IPC handler."""
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='panel-diagnostic', daemon=True)
        self.thread.start()

    def submit(self, path, value):
        with self.condition:
            self.pending = (path, value)
            self.condition.notify()

    def _write(self, path, value):
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, allow_nan=False))
        tmp.replace(path)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.pending is None:
                    return
                item, self.pending = self.pending, None
            try:
                self._write(*item)
            except Exception as error:
                self.error = type(error).__name__ + ': ' + str(error)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(.5)


class WorkerTransport(BoardTransport):
    request_cap = None

    def tick(self):
        trace=getattr(self.remote_panel,'packet_trace',None)
        started=time.monotonic()
        if trace is not None:
            trace.record('publisher_tick_begin', mode=self.mode,
                         producer_deadline_monotonic_s=self.deadline)
        try:return super().tick()
        finally:
            if trace is not None:
                trace.record('publisher_tick_return', mode=self.mode,
                             elapsed_ms=(time.monotonic()-started)*1000,
                             last_sent_monotonic_s=self.last_sent)

    def _fail(self, reason, panel=None):
        trace=getattr(self.remote_panel,'packet_trace',None)
        if trace is not None and not self.fault:
            trace.fault('publisher_fault', reason_detail=reason, mode=self.mode,
                        producer_deadline_monotonic_s=self.deadline,
                        last_sent_monotonic_s=self.last_sent)
        return super()._fail(reason, panel)

    def _send_remote(self, command):
        if self.request_cap is not None:
            self.deadline = min(self.deadline, self.request_cap)
        return super()._send_remote(command)


def serve(sock, parent_pid):
    panel = transport = None
    recent = deque(maxlen=128)
    diagnostic = None
    writer = LatestDiagnostic()

    def record(event, **fields):
        recent.append(dict(event=event, time_monotonic=time.monotonic(), **fields))
        trace = getattr(panel, 'packet_trace', None)
        if trace is not None:
            trace.record('ipc_'+event, **fields)
            if event == 'error':
                trace.fault('ipc_request_failed', op=fields.get('op'),
                            reason_detail=fields.get('reason'))

    def save():
        if diagnostic is None:
            return
        value = dict(worker_pid=os.getpid(), parent_pid=parent_pid, recent=list(recent),
                     diagnostic_write_error=writer.error)
        if transport is not None:
            with transport.lock:
                value.update(mode=transport.mode, deadline=transport.deadline,
                             fault=transport.fault, last_sent=transport.last_sent)
        if panel is not None:
            with panel.lock:
                value['link'] = copy.deepcopy(panel.timing)
                trace = getattr(panel, 'packet_trace', None)
                if trace is not None:
                    value['packet_trace'] = trace.status()
        writer.submit(diagnostic, value)

    def snapshot():
        result = {}
        if panel is not None:
            with panel.lock:
                result['panel'] = dict(data=panel.data, stamp=panel.stamp, packet=panel.packet)
        if transport is not None:
            with transport.lock:
                result['transport'] = dict(fault=transport.fault, mode=transport.mode,
                    epoch=transport.epoch, session=transport.session, deadline=transport.deadline,
                    feedback_wait=None if transport.feedback_wait is None else dict(transport.feedback_wait))
        return result

    try:
        while True:
            if not select.select([sock], [], [], .05)[0]:
                continue
            request = read_message(sock, time.monotonic()+.3)
            op, args = request['op'], request['args']
            handled_at = time.monotonic()
            record('request', op=op, request_id=request['id'],
                   age_ms=(handled_at-request['sent_at'])*1000)
            response = dict(id=request['id'], result=None)
            try:
                if op == 'open':
                    if panel is not None:
                        raise RuntimeError('Panel IPC already opened')
                    panel = BoardPanel(**args)
                    diagnostic = Path(f'/tmp/cuarm-v31-publisher-{os.getuid()}-{args["local"][1]}.json')
                    if os.getppid() != parent_pid:
                        raise RuntimeError('Panel publisher parent changed during startup')
                    with panel.lock:
                        panel.timing.update(planner_pid=parent_pid, publisher_pid=os.getpid(),
                            planner_start=process_identity(parent_pid)[1],
                            publisher_start=process_identity(os.getpid())[1],
                            publisher_sha256=publisher_fingerprint())
                elif op == 'configure':
                    if panel is None or transport is not None:
                        raise RuntimeError('Invalid panel IPC configuration phase')
                    transport = WorkerTransport(panel, **args, log=lambda s: print(s, flush=True))
                elif op == 'feedback':
                    panel.receive_with_timestamp()
                elif op == 'close':
                    if transport is not None:
                        transport.terminate()
                    if panel is not None:
                        panel.terminate()
                    write_message(sock, response, time.monotonic()+.3)
                    break
                elif op in ('start', 'begin_cycle', 'ready_to_plan', 'begin_planning', 'idle_hold', 'send'):
                    if transport is None:
                        raise RuntimeError('Panel transport not configured')
                    # Reject queued/stalled commands. Bounded planning is the only
                    # operation allowed to outlive the short producer lease.
                    origin = request['sent_at']
                    if not math.isfinite(origin) or not 0 <= time.monotonic()-origin <= REQUEST_AGE:
                        raise RuntimeError('Panel IPC request expired; nothing dispatched')
                    with transport.lock:
                        lease = transport.PLANNING_LEASE if op == 'begin_planning' else transport.PRODUCER_LEASE
                        transport.request_cap = origin+lease if op in ('begin_planning','idle_hold','send') else None
                        try:
                            response['result'] = getattr(transport, op)(**args)
                        finally:
                            transport.request_cap = None
                    if op == 'start':
                        print(f'V3.1 UDP publisher process: pid={os.getpid()}, planner_pid={parent_pid}; '
                              'bounded measured hold independent of CuRobo/GIL', flush=True)
                else:
                    raise ValueError('Unknown panel IPC operation')
            except Exception as error:
                record('error', op=op, reason=str(error))
                response['error'] = dict(type=type(error).__name__, message=str(error))
                save()
            record('handled', op=op, request_id=request['id'],
                   elapsed_ms=(time.monotonic()-handled_at)*1000)
            response.update(snapshot())
            # Never hold a transport/panel lock while writing to the parent.
            # A paused parent must not block the independent fixed-hold sender.
            reply_at = time.monotonic()
            record('reply_begin', op=op, request_id=request['id'])
            write_message(sock, response, time.monotonic()+.3)
            record('reply_return', op=op, request_id=request['id'],
                   elapsed_ms=(time.monotonic()-reply_at)*1000)
            if op in ('begin_planning','send'):
                record('commit', op=op, accepted='error' not in response)
                save()
    except (EOFError, OSError) as error:
        record('ipc_closed', reason=str(error))
    finally:
        # EOF (including killed parent) revokes the lease; do not wait 10s.
        if transport is not None:
            transport.terminate()
        if panel is not None:
            panel.terminate()
        record('closed')
        save()
        writer.close()
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    def stop(*_): raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    serve(socket.socket(fileno=args.fd), args.parent_pid)


if __name__ == '__main__':
    main()
