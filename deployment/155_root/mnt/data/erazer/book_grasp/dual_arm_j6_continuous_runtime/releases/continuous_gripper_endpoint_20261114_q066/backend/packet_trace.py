"""Diagnostic-only packet journal. No payloads, credentials or control decisions.

Callers only enqueue bounded records. File creation, JSON encoding, rotation and
flushing run in the writer. send_return means the OS accepted a datagram, not
that a NIC transmitted it. Monotonic clocks are comparable on ONE host only.
"""
import atexit
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def packet_identity(packet, raw):
    """Only identifiers: never retain the session credential or motion payload."""
    result = dict(wire_sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if isinstance(packet, dict):
        session = packet.get('session')
        if isinstance(session, str):
            result['session_tag'] = hashlib.sha256(session.encode()).hexdigest()[:12]
        for key in ('sequence', 'target_id'):
            if type(packet.get(key)) is int:
                result[key] = packet[key]
        result['kind'] = ('state' if packet.get('kind') == 'state' else
                          'cartesian' if packet.get('kind') == 'cartesian' else 'native')
        transit = packet.get('transit', {})
        if isinstance(transit, dict) and type(transit.get('feedback_sequence')) is int:
            result['feedback_sequence'] = transit['feedback_sequence']
    return result


class PacketTrace:
    """Bounded async journal with an immutable pre/post-fault capture per fault.

    Up to 4096 queued records + 4096 history records; four 8MiB rolling files,
    eight fault captures per process. A slow/full/broken disk drops diagnostics
    (with counters), never blocks dispatch or renews any command lease.
    """
    def __init__(self, directory, side, *, capacity=4096, history=4096,
                 max_bytes=8*1024*1024, rotations=4, post_seconds=5., fault_limit=8):
        self.pid = os.getpid()
        self.side = side
        self.run_id = f'{side}-{self.pid}-{time.time_ns()}'
        self.directory = Path(directory)/self.run_id
        self.capacity = capacity
        self.history = deque(maxlen=history)
        self.pending = deque()
        self.faults = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = False
        self.serial = self.dropped = self.written = 0
        self.error = None
        self.fault_count = 0
        self.max_bytes, self.rotations = max_bytes, rotations
        self.post_seconds, self.fault_limit = post_seconds, fault_limit
        self.thread = threading.Thread(target=self._writer, name='packet-trace', daemon=True)
        self.thread.start()
        atexit.register(self.close)
        self.record('trace_started', schema=1, pid=self.pid,
                    clock_policy='same_host_monotonic_only',
                    send_semantics='sendto_return_is_os_acceptance_not_wire_tx')

    def record(self, event, **fields):
        if self.closed:
            return
        # No control thread waits for the writer's queue lock or disk.
        if not self.lock.acquire(False):
            self.dropped += 1
            return
        try:
            self.serial += 1
            row = dict(fields, event=event, side=self.side, index=self.serial,
                       monotonic_s=time.monotonic(), unix_ns=time.time_ns())
            self.history.append(row)
            if len(self.pending) < self.capacity:
                self.pending.append(row)
            else:
                self.dropped += 1
        finally:
            self.lock.release()
        self.wake.set()

    def fault(self, reason, **fields):
        self.record('fault', reason=reason, **fields)
        # Only a short bounded memory copy; the writer never holds this lock
        # during file I/O. Keep the first faults instead of overwriting them.
        with self.lock:
            if self.fault_count >= self.fault_limit or self.closed:
                return
            self.fault_count += 1
            self.faults.append((self.fault_count, self.serial, time.monotonic(),
                                list(self.history), str(reason)))
        self.wake.set()

    def status(self):
        return dict(directory=str(self.directory), records=self.serial,
                    written=self.written, dropped=self.dropped,
                    fault_captures=self.fault_count, error=self.error)

    @staticmethod
    def _line(row):
        return json.dumps(row, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))+'\n'

    def _writer(self):
        stream = None
        captures = []
        size = 0
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            path = self.directory/'packets.jsonl'
            stream = path.open('w', buffering=65536)
            next_flush = time.monotonic()+.25
            while True:
                self.wake.wait(.05)
                self.wake.clear()
                with self.lock:
                    batch = [self.pending.popleft() for _ in range(min(128, len(self.pending)))]
                    faults = list(self.faults)
                    self.faults.clear()
                    done = self.closed and not self.pending
                for number, index, stamp, history, reason in faults:
                    capture = (self.directory/f'fault-{number:02d}.jsonl').open('x')
                    capture.write(self._line(dict(event='fault_capture', reason=reason,
                        fault_monotonic_s=stamp, through_index=index, **self.status())))
                    for row in history:
                        capture.write(self._line(row))
                    captures.append((capture, index, stamp+self.post_seconds))
                for row in batch:
                    line = self._line(row)
                    if size+len(line.encode()) > self.max_bytes:
                        stream.close()
                        for n in range(self.rotations-1, 0, -1):
                            src = path if n == 1 else self.directory/f'packets.{n-1}.jsonl'
                            if src.exists(): src.replace(self.directory/f'packets.{n}.jsonl')
                        stream = path.open('w', buffering=65536)
                        size = 0
                    stream.write(line)
                    size += len(line.encode())
                    self.written += 1
                    for capture, index, until in captures:
                        if row['index'] > index and row['monotonic_s'] <= until:
                            capture.write(line)
                now = time.monotonic()
                if now >= next_flush or faults or done:
                    stream.flush()
                    for capture, _, _ in captures: capture.flush()
                    next_flush = now+.25
                # Drain queued post-fault records even if disk was slow.
                last_stamp = batch[-1]['monotonic_s'] if batch else now
                for item in captures[:]:
                    if last_stamp > item[2] or done:
                        item[0].write(self._line(dict(event='capture_closed', **self.status())))
                        item[0].close()
                        captures.remove(item)
                if done: break
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            for capture, _, _ in captures:
                try: capture.close()
                except OSError: pass
            if stream is not None:
                try: stream.close()
                except OSError: pass

    def close(self):
        if self.closed:
            return
        self.record('trace_closed', **self.status())
        self.closed = True
        self.wake.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(.5)


def traced_send(trace, sock, raw, destination, packet, *, prepared_at, **fields):
    """Wrap exactly one existing sendto; logging never retries or changes it."""
    if trace is None:
        return sock.sendto(raw, destination)
    identity = packet_identity(packet, raw)
    started = time.monotonic()
    trace.record('send_begin', prepared_monotonic_s=prepared_at,
                 send_started_monotonic_s=started,
                 prepare_ms=(started-prepared_at)*1000, **identity, **fields)
    try:
        result = sock.sendto(raw, destination)
    except Exception as error:
        ended = time.monotonic()
        trace.record('send_error', send_started_monotonic_s=started,
                     send_return_monotonic_s=ended, syscall_ms=(ended-started)*1000,
                     error_type=type(error).__name__, **identity)
        trace.fault('sendto_failed', error_type=type(error).__name__)
        raise
    ended = time.monotonic()
    trace.record('send_return', send_started_monotonic_s=started,
                 send_return_monotonic_s=ended, syscall_ms=(ended-started)*1000,
                 **identity)
    return result
