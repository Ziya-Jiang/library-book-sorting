"""Bounded, credential-free UDP and control-loop timing for the V3.1 bridges.

Kernel timestamps use CLOCK_REALTIME; queue age is calculated on this host only,
then translated to monotonic time. No cross-machine clock subtraction is used.
No background receive thread, file I/O, or motion is performed here.
"""
from collections import deque
import copy
import json
import math
import socket
import struct
import time

from .v31_wire import LEASE_S

# Linux SO_TIMESTAMPING_NEW, software RX generation + reporting only.
# Do not also enable SO_TIMESTAMP[NS]: those can synthesize a recvmsg-time
# timestamp when the actual RX stamp is missing (kernel timestamping docs).
TIMESTAMPING = getattr(socket, "SO_TIMESTAMPING_NEW", 65)
RX_TIMESTAMP_FLAGS = (1 << 3) | (1 << 4)
TIMESPEC = struct.Struct("=qq")
ANCILLARY_SIZE = TIMESPEC.size * 3
CONTROL_LOOP_LIMIT_S = LEASE_S


def arrival_stamp(ancillary, flags, processed, wall_ns):
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise ValueError("truncated command or kernel timestamp")
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == TIMESTAMPING:
            if len(value) < ANCILLARY_SIZE:
                raise ValueError("short kernel timestamp")
            sec, ns = TIMESPEC.unpack(value[:TIMESPEC.size])
            age = (wall_ns-(sec*1_000_000_000+ns))/1e9
            if sec <= 0 or not 0 <= ns < 1_000_000_000 or age < 0:
                raise ValueError("invalid kernel timestamp or wall clock reversal")
            return processed-age, age, sec*1_000_000_000+ns
    raise ValueError("missing kernel arrival timestamp")


class ControlLoopTiming:
    def __init__(self):
        self.started = self.previous_start = None
        self.last_phase = None
        self.current = {}
        self.recent = deque(maxlen=8)

    def begin(self, requested, acquired):
        self.previous_start, self.started = self.started, acquired
        self.last_phase = acquired
        self.current = dict(start_monotonic_s=acquired,
                            lock_wait_ms=max(0., acquired-requested)*1000,
                            start_gap_ms=None if self.previous_start is None else
                            (acquired-self.previous_start)*1000, phases_ms={})

    def mark(self, name, now=None):
        now = time.monotonic() if now is None else now
        self.current["phases_ms"][name] = max(0., now-self.last_phase)*1000
        self.current["last_completed_phase"] = name
        self.last_phase = now

    def end(self, now=None):
        now = time.monotonic() if now is None else now
        self.current["duration_ms"] = max(0., now-self.started)*1000
        self.recent.append(copy.deepcopy(self.current))

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        current = copy.deepcopy(self.current)
        if self.started is not None:
            current["elapsed_ms"] = max(0., now-self.started)*1000
            current["unfinished_phase_ms"] = max(0., now-self.last_phase)*1000
        return dict(current=current, recent=list(self.recent))


class PacketTiming:
    def __init__(self):
        self.current = None
        self.last_valid_arrival = self.last_valid_processed = None
        self.last_tick = None
        self.tick_gap = None
        self.tick_enabled = False
        self.recent = deque(maxlen=8)
        self.max_queue_ms = 0.
        self.max_tick_gap_ms = 0.
        self.previous_clock_offset = None
        self.previous_clock_uncertainty = 0.

    def reset_epoch(self):
        self.last_valid_arrival = self.last_valid_processed = None
        self.last_tick = None
        self.tick_gap = None
        self.tick_enabled = False
        self.current = None
        self.recent.clear()

    def received(self, ancillary, flags, peer_matches, now, wall_ns, clock_offset=None,
                 clock_uncertainty_s=0.):
        self.current = dict(dequeued_monotonic_s=now, peer_matches=peer_matches,
                            outcome="timestamp_unverified")
        self.recent.append(self.current)
        try:
            arrival, age, kernel_ns = arrival_stamp(ancillary, flags, now, wall_ns)
            offset = wall_ns/1e9-now if clock_offset is None else clock_offset
            uncertainty = clock_uncertainty_s+self.previous_clock_uncertainty
            if self.previous_clock_offset is not None and abs(offset-self.previous_clock_offset) > .005+uncertainty:
                raise ValueError("wall/monotonic clock offset changed during command reception")
            self.previous_clock_offset = offset
            self.previous_clock_uncertainty = clock_uncertainty_s
            self.current.update(kernel_realtime_ns=kernel_ns, arrival_monotonic_s=arrival,
                                queue_ms=age*1000,
                                clock_sample_uncertainty_ms=clock_uncertainty_s*1000,
                                outcome="dequeued")
            self.max_queue_ms = max(self.max_queue_ms, age*1000)
        except ValueError as error:
            self.current.update(outcome="timestamp_invalid", reason=str(error))
            raise

    def packet(self, packet, outcome):
        if self.current is None:
            return
        seq = packet.get("sequence")
        self.current.update(sequence=seq if type(seq) is int else None,
                            kind="cartesian" if packet.get("kind") == "cartesian" else "native",
                            outcome=outcome)
        echo = packet.get('echo')
        if type(echo) in (int, float) and math.isfinite(echo):
            self.current['echo_age_at_dequeue_ms'] = (self.current['dequeued_monotonic_s']-echo)*1000
            arrival = self.current.get('arrival_monotonic_s')
            if arrival is not None:
                self.current['echo_age_at_arrival_ms'] = (arrival-echo)*1000
        # Optional sender measurements are diagnostic only. They never take
        # part in freshness, session, collision, sequence, or lease decisions.
        transit = packet.get('transit')
        if isinstance(transit, dict):
            values = {k: v for k, v in transit.items()
                      if k in ('pc_residence_ms', 'pc_prepare_ms', 'pc_feedback_queue_ms')
                      and type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 60000}
            self.current.update(values)
            seq = transit.get('feedback_sequence')
            if type(seq) is int and seq >= 1:
                self.current['feedback_sequence'] = seq
            if 'pc_residence_ms' in values and 'echo_age_at_arrival_ms' in self.current:
                self.current['outside_pc_residence_ms'] = (
                    self.current['echo_age_at_arrival_ms']-values['pc_residence_ms'])

    def check_fresh(self, now):
        if self.current is None or "arrival_monotonic_s" not in self.current:
            raise ValueError("command lacks verified receive timing")
        if now-self.current["arrival_monotonic_s"] > LEASE_S:
            self.current.update(outcome="arrival_expired", rejected_monotonic_s=now,
                                processing_ms=(now-self.current["dequeued_monotonic_s"])*1000)
            raise ValueError("command kernel arrival expired; queue/processing delay exceeds 250ms")

    def accepted(self, now):
        self.check_fresh(now)
        self.current.update(outcome="accepted", validated_monotonic_s=now,
                            processing_ms=(now-self.current["dequeued_monotonic_s"])*1000)
        self.last_valid_arrival = self.current["arrival_monotonic_s"]
        self.last_valid_processed = now

    def snapshot(self, now):
        return dict(last_valid_arrival_monotonic_s=self.last_valid_arrival,
                    last_valid_processed_monotonic_s=self.last_valid_processed,
                    valid_arrival_age_ms=None if self.last_valid_arrival is None else
                    (now-self.last_valid_arrival)*1000,
                    valid_processing_age_ms=None if self.last_valid_processed is None else
                    (now-self.last_valid_processed)*1000,
                    tick_gap_ms=None if self.tick_gap is None else self.tick_gap*1000,
                    max_tick_gap_ms=self.max_tick_gap_ms, max_queue_ms=self.max_queue_ms,
                    recent=copy.deepcopy(list(self.recent)))


class TimedPlannerBridge:
    """Mixin shared by native, board and full-visual bridges."""
    def _timing(self):
        if not hasattr(self, "receive_timing"):
            self.receive_timing = PacketTiming()
        return self.receive_timing

    def configure(self, data):
        super().configure(data)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, TIMESTAMPING, RX_TIMESTAMP_FLAGS)
            self.receive_timing = PacketTiming()
        except Exception:
            self.disconnect()
            raise

    def _reset_timing_epoch(self):
        self._timing().reset_epoch()

    def _receive_command(self):
        raw, ancillary, flags, sender = self.sock.recvmsg(60001, socket.CMSG_SPACE(ANCILLARY_SIZE))
        now = time.monotonic()
        mapping_mono_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        mapping_end_ns = time.monotonic_ns()
        self._timing().received(ancillary, flags,
                               sender == (self.settings["ip"], self.settings["port"]),
                               now, wall_ns, (wall_ns-(mapping_mono_ns+mapping_end_ns)//2)/1e9,
                               (mapping_end_ns-mapping_mono_ns)/2e9)
        return raw, sender

    def _begin_timed_tick(self, now):
        t = self._timing()
        previous_enabled = t.tick_enabled
        t.tick_gap = None if t.last_tick is None else now-t.last_tick
        t.last_tick, t.tick_enabled = now, self.enabled
        if t.tick_gap is not None:
            t.max_tick_gap_ms = max(t.max_tick_gap_ms, t.tick_gap*1000)
        if not self.enabled:
            return
        self._check_timing_deadlines(now, check_gap=previous_enabled)

    def _check_timing_deadlines(self, now, check_gap=False):
        t = self._timing()
        loop = getattr(self.c, "loop_timing", None)
        cycle_elapsed = None if loop is None or loop.started is None else now-loop.started
        cycle_gap = None if loop is None else loop.current.get("start_gap_ms")
        loop_timeout = ((check_gap and t.tick_gap is not None and t.tick_gap > CONTROL_LOOP_LIMIT_S)
                        or (check_gap and cycle_gap is not None and cycle_gap > CONTROL_LOOP_LIMIT_S*1000)
                        or (cycle_elapsed is not None and cycle_elapsed > CONTROL_LOOP_LIMIT_S))
        # The old processing-time lease remains a conservative backstop.
        arrival = t.last_valid_arrival
        lease_timeout = self.command_seen and (
            now-self.last_rx > LEASE_S or (arrival is not None and now-arrival > LEASE_S))
        self.timing_timeout_flags = dict(control_loop_timeout=bool(loop_timeout),
                                         valid_command_lease_timeout=bool(lease_timeout),
                                         command_limit_s=LEASE_S, control_loop_limit_s=CONTROL_LOOP_LIMIT_S)
        if loop_timeout or lease_timeout:
            labels = []
            if loop_timeout:
                labels.append("control loop deadline expired")
            if lease_timeout:
                labels.append("planner lease expired")
            self.stop_fault("V3.1 "+"; ".join(labels)+"; reconnect/re-arm required")

    def check_control_deadline(self):
        if self.enabled:
            self._check_timing_deadlines(time.monotonic())

    def _accept_timed_command(self):
        # Called after validation; failure cancels staged targets before RT send.
        self._timing().accepted(time.monotonic())

    def command_receipt(self):
        """Small feedback receipt; no full paths, tokens or unbounded history."""
        t = self._timing()
        return dict(last_packet=dict(t.current) if t.current else None,
                    accepted_sequence=self.gate.sequence,
                    discarded_count=self.gate.discarded,
                    expired_echo_count=self.gate.expired_echoes)

    def timing_snapshot(self, peek=False):
        now = time.monotonic()
        result = dict(packet=self._timing().snapshot(now),
                      timeouts=dict(getattr(self, "timing_timeout_flags", {})),
                      note="arrival age is last validated packet, not proof of an empty kernel queue")
        loop = getattr(self.c, "loop_timing", None)
        if loop is not None:
            result["control_loop"] = loop.snapshot(now)
            if self.enabled and loop.started is not None and now-loop.started > CONTROL_LOOP_LIMIT_S:
                result["timeouts"]["control_loop_timeout"] = True
        if peek and self.sock is not None:
            # Peek is evidence only: never execute or renew a lease from it.
            try:
                raw, ancillary, flags, sender = self.sock.recvmsg(
                    60001, socket.CMSG_SPACE(ANCILLARY_SIZE), socket.MSG_PEEK | socket.MSG_DONTWAIT)
                sampled = time.monotonic()
                arrival, age, _ = arrival_stamp(ancillary, flags, sampled, time.time_ns())
                seq = None
                try:
                    packet = json.loads(raw)
                    if isinstance(packet, dict) and type(packet.get("sequence")) is int:
                        seq = packet["sequence"]
                except (ValueError, UnicodeError):
                    pass
                result["queued_head"] = dict(arrival_monotonic_s=arrival, age_ms=age*1000,
                                             sequence=seq, validated=False,
                                             peer_matches=sender == (self.settings["ip"], self.settings["port"]))
            except BlockingIOError:
                result["queued_head"] = None
            except (OSError, ValueError) as exc:
                result["queued_head_error"] = str(exc)
        return result
