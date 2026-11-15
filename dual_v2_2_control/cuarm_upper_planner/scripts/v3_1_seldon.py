"""Explicit network profile for the isolated Seldon V3.1 cockpit."""
import ctypes
import copy
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import select
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT/'seldon_v31'
sys.path.insert(0, str(PROFILE))
from v31_wire import decode, encode, ECHO_AGE_S, command as decode_command
from packet_trace import PacketTrace, packet_identity, traced_send
from v3_1_panel_transport import PanelTransport, ControlSessionCancelled
from v3_1_direct_servo import DirectVisualServo
from v3_1_execution import ExecutionGuard
from v3_1_motion_bridge import SnapshotUnavailable

DISPATCH_BUDGET_S = .10


def endpoints(host, bind='0.0.0.0'):
    remote = ipaddress.IPv4Address(host)
    local = ipaddress.IPv4Address(bind)
    if remote.is_unspecified or remote.is_multicast or remote.is_loopback or str(remote) == '255.255.255.255':
        raise ValueError('Specify the robot LAN IPv4 address')
    if local.is_multicast:
        raise ValueError('Invalid planner bind address')
    return (str(local), 30241), (str(remote), 30240)


def expected_contract():
    return {name: item['sha256'] for name, item in
            json.loads((PROFILE/'contract.json').read_text()).items()}


def check_profile():
    expected = expected_contract()
    for name in ('interpolation.cpp', 'interpolation.h'):
        path = PROFILE/'rt_reference/cuarm_rt_control/cuarm_robotics/src'/name
        if hashlib.sha256(path.read_bytes().rstrip(b'\n')).hexdigest() != expected[name]:
            raise ValueError('Remote interpolation snapshot mismatch: '+name)
    return expected


def check_robot_config(config):
    expected = expected_contract()
    urdf = Path(config.robot_cfg['kinematics']['urdf_path'])
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != expected['model']:
        raise ValueError('155 V2 robot model differs from planner URDF')
    if config.arm_joint_size != [7, 7] or config.gripper_joint_size != [6, 1]:
        raise ValueError('155 requires two 7-joint arms, left hand 6 and right gripper 1')


def normalize_hold(command):
    """Canonicalize remote single-point fields without changing waypoint paths."""
    result = copy.deepcopy(command)
    mode = getattr(result.arm_target_mode, 'value', result.arm_target_mode)
    raw_hold = (mode == 2 and result.arm_waypoint_size == 2 and
                result.arm_waypoint_command == [result.arm_joint_command, result.arm_joint_command])
    if raw_hold:
        result.arm_target_mode = type(result.arm_target_mode)(0)
        result.interpolation_acceleration_time = .2
        mode = 0
    if mode in (0, 2):
        result.arm_waypoint_size = 0
        result.arm_waypoint_dt = 0.0
        result.arm_waypoint_command = []
    if mode == 0 and result.interpolation_acceleration_time == .2:
        result.no_interpolation_max_velocity_ratio = min(.1, result.no_interpolation_max_velocity_ratio)
    return result


class FeedbackBusy(RuntimeError):
    """Another bounded receive transaction is still in progress."""

class FeedbackBacklog(FeedbackBusy):
    """A bounded drain has not caught up; no cached state is dispatchable."""


class RemotePanel:
    TRACE_PACKETS = False  # Enabled only by the book board Planner subclass.
    RECOVER_SESSION_CANCELLATION = False
    TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)
    RECEIVE_BATCH = 64
    MAX_REFRESH_BATCHES = 8
    REFRESH_BUDGET = DISPATCH_BUDGET_S

    def __init__(self, local, remote, timeout=10):
        from cuarm_udp import CuarmUdp
        self.codec = CuarmUdp.__new__(CuarmUdp)
        self.remote = remote
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._closed = False
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.receive_lock = threading.Lock()
        self.data = self.stamp = self.packet = None
        self.sequence = 0
        self.last_rt = None
        self.error = None
        self.contract = check_profile()
        self.rx_stamp = None
        self.timing = dict(received=0, sent=0, coalesced=0, backlog_batches=0,
                           max_kernel_queue_ms=0., max_pack_ms=0., max_send_gap_ms=0.)
        self.last_send = None
        self.next_report = 0.
        self.timing_path = Path(f'/tmp/cuarm-v31-seldon-link-{os.getuid()}-{local[1]}.json')
        self.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.packet_trace = None
        if self.TRACE_PACKETS:
            directory = os.environ.get('BOOK_PACKET_TRACE_DIR') or str(
                ROOT.parent.parent/'robot_cockpit/launcher/runtime/book_grasp/packet_trace')
            self.packet_trace = PacketTrace(directory, 'pc')
            print('PC逐包日志：'+str(self.packet_trace.directory), flush=True)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, self.TIMESTAMPNS, 1)
            self.sock.bind(local)
            self.thread = threading.Thread(target=self._receive, daemon=True)
            self.thread.start()
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                if self.error:
                    raise ValueError(self.error)
                if self.packet is not None:
                    break
                time.sleep(.01)
            else:
                raise RuntimeError(
                    f'No V3.1 cockpit handshake within {timeout:g}s. Expected feedback '
                    f'{remote[0]}:{remote[1]} -> {local[0]}:{local[1]}. '
                    'Check fresh RT feedback in the cockpit, then apply this PC reachable IPv4 '
                    f'and port {local[1]} as the Planner destination (never 0.0.0.0). '
                    'For the board Planner keep the matching server_board/server_board_visual backend; '
                    'do not switch it to server_v31. Leave Planner control disabled until handshake succeeds.')
        except BaseException:
            self.terminate()
            raise

    def _read_batch(self, timeout=.05):
        self._batch_full = False
        if timeout and not select.select([self.sock], [], [], timeout)[0]:
            raise socket.timeout()
        result = []
        for attempt in range(self.RECEIVE_BATCH):
            try:
                raw, ancillary, flags, sender = self.sock.recvmsg(60001, socket.CMSG_SPACE(struct.calcsize('@ll')))
            except BlockingIOError:
                return result
            if sender != self.remote:
                continue
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise ValueError('Truncated V3.1 feedback or kernel timestamp')
            received = None
            now = time.monotonic()
            wall_ns = time.time_ns()
            for level, kind, value in ancillary:
                if level == socket.SOL_SOCKET and kind == self.TIMESTAMPNS:
                    seconds, nanoseconds = struct.unpack('@ll', value[:struct.calcsize('@ll')])
                    queue_age = (wall_ns-seconds*1_000_000_000-nanoseconds)/1e9
                    if not 0 <= nanoseconds < 1_000_000_000 or queue_age < 0:
                        raise ValueError('Invalid V3.1 kernel arrival timestamp')
                    received = now-queue_age
                    break
            if received is None:
                raise ValueError('V3.1 kernel arrival timestamp missing')
            trace = getattr(self, 'packet_trace', None)
            if trace is not None:
                trace.record('receive', wire_sha256=hashlib.sha256(raw).hexdigest(),
                             size_bytes=len(raw), arrival_monotonic_s=received,
                             dequeued_monotonic_s=now, queue_ms=queue_age*1000,
                             kernel_realtime_ns=seconds*1_000_000_000+nanoseconds)
            result.append((raw, sender, received, queue_age))
        with self.lock:
            self.timing['backlog_batches'] += 1
        # A full chunk does not prove loss/corruption or even that another
        # packet remains. Continue draining under the refresh time budget.
        self._batch_full = True
        return result

    def _report(self):
        now = time.monotonic()
        with self.lock:
            if now < self.next_report:
                return
            self.next_report = now+1.
            status = dict(self.timing, pid=os.getpid(), updated_monotonic=now,
                          remote=list(self.remote), error=self.error,
                          adapter_sha256=self.adapter_sha256, echo_age_s=ECHO_AGE_S,
                          dispatch_budget_s=DISPATCH_BUDGET_S)
            trace = getattr(self, 'packet_trace', None)
            if trace is not None:
                status['packet_trace'] = trace.status()
        temporary = self.timing_path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            temporary.write_text(json.dumps(status, allow_nan=False))
            temporary.replace(self.timing_path)
        except OSError as error:
            print('V3.1 link diagnostics unavailable: '+str(error), file=sys.stderr)

    def _consume_batch(self, batch):
        from cuarm_udp import CuarmUdp
        from cuarm_state import PanelRemoteState
        with self.lock:
            candidate, data, stamp, identity = self.packet, self.data, self.stamp, self.last_rt
            received = self.rx_stamp
        accepted = 0
        for raw, sender, arrival, queue_age in batch:
            if sender != self.remote:
                continue
            with self.lock:
                self.timing['received'] += 1
                self.timing['max_kernel_queue_ms'] = max(self.timing['max_kernel_queue_ms'], queue_age*1000)
                self.timing['kernel_queue_ms'] = queue_age*1000
            try:
                packet = decode(raw)
            except json.JSONDecodeError as error:
                if raw.startswith((b'0#2#7#7#', b'1#2#7#7#')):
                    raise ValueError(
                        '155 is sending legacy PanelState, not V3.1 JSON. '
                        'On 155 switch seldon_arm.server to seldon_arm.server_v31 '
                        'after safely ending motion, then apply the Planner address. '
                        'Legacy feedback rejected; RAW requires the V3.1 bridge.') from error
                raise ValueError('Invalid V3.1 JSON feedback from configured robot endpoint') from error
            if packet.get('kind') != 'state' or packet.get('contract') != self.contract:
                raise ValueError('155 V3.1 execution contract mismatch; re-audit RT/calibration before connecting')
            if packet.get('echo_age_s') != ECHO_AGE_S:
                raise ValueError(f'155 V3.1 echo policy mismatch: expected {ECHO_AGE_S*1000:g}ms. '
                                 'Reload the updated 155 arm backend after confirming the robot is stationary; '
                                 'do not enable control with mixed policies.')
            session, sequence = packet.get('session'), packet.get('sequence')
            if not isinstance(session, str) or len(session) != 32 or type(sequence) is not int or sequence < 1:
                raise ValueError('Invalid V3.1 state identity')
            timestamp, echo = packet.get('rt_timestamp'), packet.get('echo')
            if type(timestamp) is not int or timestamp <= 0 or type(echo) not in (int, float) or not math.isfinite(echo):
                raise ValueError('Invalid V3.1 state timestamp')
            limits = packet.get('speed_limits')
            if not isinstance(limits, list) or len(limits) != 2 or any(
                    len(row) != 7 or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in row)
                    for row in limits):
                raise ValueError('Invalid remote velocity limits')
            decoded = CuarmUdp.unpack_panel_state(self.codec, packet['state'])
            if decoded is None or decoded.arm_joint_size != [7, 7] or decoded.gripper_joint_size != [6, 1]:
                raise ValueError('Invalid remote robot layout')
            if not all(math.isfinite(value) for rows in
                       (decoded.arm_joint_position, decoded.arm_joint_velocity, decoded.gripper_joint_position)
                       for row in rows for value in row):
                raise ValueError('Invalid remote feedback')
            decoded._seldon_session = session
            trace = getattr(self, 'packet_trace', None)
            if trace is not None:
                trace.record('feedback_processed', **packet_identity(packet, raw),
                    arrival_monotonic_s=arrival,
                    dequeued_monotonic_s=arrival+queue_age,
                    processing_ms=(time.monotonic()-arrival-queue_age)*1000,
                    queue_ms=queue_age*1000, rt_timestamp=timestamp,
                    rt_advanced=(session, timestamp) != identity,
                    reordered=bool(candidate and session == candidate['session'] and
                                   sequence <= candidate['sequence']))
            if candidate and session == candidate['session'] and sequence <= candidate['sequence']:
                continue
            if candidate and session != candidate['session']:
                identity = None
            if identity and session == identity[0] and timestamp < identity[1]:
                raise ValueError('Remote RT timestamp moved backwards')
            next_identity = (session, timestamp)
            if next_identity != identity:
                stamp = arrival
            identity = next_identity
            candidate, data, received = packet, decoded, arrival
            candidate_queue_ms = queue_age*1000
            receipt = packet.get('command_receipt')
            if isinstance(receipt, dict):
                row = receipt.get('last_packet')
                if isinstance(row, dict) and type(row.get('sequence')) is int:
                    # Only small, bounded diagnostic fields are copied. Never
                    # let an ACK modify send eligibility or command ownership.
                    sample = {k: row[k] for k in (
                        'sequence', 'feedback_sequence', 'queue_ms', 'processing_ms',
                        'echo_age_at_arrival_ms', 'echo_age_at_dequeue_ms',
                        'pc_residence_ms', 'pc_prepare_ms', 'pc_feedback_queue_ms',
                        'outside_pc_residence_ms') if k in row
                        and type(row[k]) in (int, float) and math.isfinite(row[k])}
                    for k in ('outcome', 'discard_reason', 'kind'):
                        if isinstance(row.get(k), str):
                            sample[k] = row[k][:80]
                    sample['session_tag'] = hashlib.sha256(session.encode()).hexdigest()[:12]
                    with self.lock:
                        history = self.timing.setdefault('command_receipts', [])
                        if not history or (history[-1].get('session_tag'), history[-1].get('sequence')) != (sample['session_tag'], sample['sequence']):
                            history.append(sample)
                            del history[:-64]
                        self.timing['last_command_receipt'] = sample
            accepted += 1
        if accepted:
            with self.lock:
                if self.packet and candidate['session'] != self.packet['session']:
                    # Native and Cartesian sends share a process-wide counter.
                    # Receiving a session change must never rewind that counter.
                    self.last_send = None
                if data.state == PanelRemoteState.STOP:
                    self.last_send = None
                self.packet, self.data, self.stamp = candidate, data, stamp
                self.last_rt, self.rx_stamp = identity, received
                self.timing['valid_feedback_queue_ms'] = candidate_queue_ms
                self.timing['coalesced'] += max(0, accepted-1)

    def _wait_readable(self):
        return bool(select.select([self.sock], [], [], .05)[0])

    def _refresh_feedback(self, timeout=REFRESH_BUDGET):
        if self._closed or self.error:
            return
        started = time.monotonic()
        with self.lock:
            previous_age = started-self.stamp if self.stamp is not None else None
        if not self.receive_lock.acquire(timeout=timeout):
            raise FeedbackBusy('V3.1 feedback receiver busy; nothing sent')
        try:
            if self._closed or self.error:
                return
            drained = False
            for _ in range(self.MAX_REFRESH_BATCHES):
                self._batch_full = False
                batch = self._read_batch(0)
                if batch is None:
                    break
                if batch:
                    # Validate every chunk, including STOP/session/RT identity.
                    # Keep kernel arrival times, never the time of this drain.
                    self._consume_batch(batch)
                if not self._batch_full:
                    drained = True
                    break
                if time.monotonic()-started >= timeout:
                    break
            if not drained:
                with self.lock:
                    self.timing['bounded_drain_waits'] = self.timing.get('bounded_drain_waits', 0)+1
                raise FeedbackBacklog('V3.1 feedback drain still pending; nothing sent')
            elapsed = time.monotonic()-started
            with self.lock:
                self.timing['max_refresh_ms'] = max(self.timing.get('max_refresh_ms', 0.), elapsed*1000)
                if previous_age is not None and previous_age > ECHO_AGE_S:
                    self.timing['max_cached_feedback_age_ms'] = max(
                        self.timing.get('max_cached_feedback_age_ms', 0.), previous_age*1000)
                    if self.stamp is not None and 0 <= time.monotonic()-self.stamp <= ECHO_AGE_S:
                        self.timing['cache_catchups'] = self.timing.get('cache_catchups', 0)+1
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            self.error = str(error)
            raise RuntimeError(self.error) from error
        finally:
            self.receive_lock.release()

    def _receive(self):
        while not self._closed:
            try:
                if self._wait_readable():
                    self._refresh_feedback()
                self._report()
            except socket.timeout:
                continue
            except FeedbackBusy:
                self._report()
                continue
            except OSError as error:
                if not self._closed:
                    self.error = str(error)
                    self._trace_fault('feedback_receive_error', error_type=type(error).__name__)
                return
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                self.error = str(error)
                self._trace_fault('feedback_receive_error', error_type=type(error).__name__)
                self.next_report = 0.
                self._report()
                return

    def receive_with_timestamp(self):
        try:
            self._refresh_feedback()
        except FeedbackBacklog:
            # Startup can wait; active PanelTransport still sees unavailable
            # feedback and latches its original stop. Do not expose a partial
            # drain or permanently kill reception of subsequent fresh states.
            return None, None
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.data, self.stamp

    def receive(self, max_age_s=None):
        data, stamp = self.receive_with_timestamp()
        if stamp is None or max_age_s is not None and time.monotonic()-stamp > max_age_s:
            return None
        return data

    def transit_sample(self, started, now):
        """Durations on this PC only; board echo remains the original stamp."""
        return dict(feedback_sequence=self.packet['sequence'],
                    pc_residence_ms=(now-self.rx_stamp)*1000,
                    pc_prepare_ms=(now-started)*1000,
                    pc_feedback_queue_ms=self.timing.get('valid_feedback_queue_ms', 0.))

    def _trace_fault(self, reason, **fields):
        trace = getattr(self, 'packet_trace', None)
        if trace is not None:
            trace.fault(reason, **fields)

    def _check_dispatch(self, *, now, started, deadline, session, phase,
                        check_rx=False, require_deadline=False):
        """Called under lock; distinguish cancellation from real timing faults."""
        from cuarm_state import PanelRemoteState
        age = now-self.stamp if self.stamp is not None else math.inf
        rx_age = now-self.rx_stamp if self.rx_stamp is not None else math.inf
        current_session = (self.packet or {}).get('session')
        reasons = []
        if self._closed: reasons.append('closed')
        if self.error: reasons.append('receiver_error')
        if not 0 <= age <= ECHO_AGE_S: reasons.append('stale_feedback')
        if check_rx and not 0 <= rx_age <= ECHO_AGE_S: reasons.append('stale_arrival')
        if now-started > DISPATCH_BUDGET_S: reasons.append('dispatch_budget')
        if deadline is not None and now > deadline: reasons.append('producer_deadline')
        if require_deadline and deadline is None: reasons.append('missing_deadline')
        if current_session != session: reasons.append('session_changed')
        if self.data is not None and self.data.state == PanelRemoteState.STOP: reasons.append('STOP')
        if not reasons: return
        tag = lambda value: hashlib.sha256(value.encode()).hexdigest()[:12] if value else None
        finite = lambda value: value if value is not None and math.isfinite(value) else None
        details = dict(phase=phase, reasons=reasons, feedback_age_s=finite(age),
                       arrival_age_s=finite(rx_age), dispatch_seconds=now-started,
                       deadline_remaining_s=finite(None if deadline is None else deadline-now),
                       expected_session_tag=tag(session), observed_session_tag=tag(current_session))
        self._trace_fault('dispatch_rejected', **details)
        message = ('V3.1 dispatch rejected ('+', '.join(r.replace('_', ' ') for r in reasons)+
                   '); nothing sent: '+json.dumps(details, ensure_ascii=False))
        if self.RECOVER_SESSION_CANCELLATION and set(reasons) <= {'session_changed', 'STOP'}:
            raise ControlSessionCancelled(message)
        raise RuntimeError(message)

    def send_strict(self, command, deadline=None, session=None):
        from cuarm_udp import CuarmUdp
        from cuarm_state import PanelRemoteState
        with self.send_lock:
            started = time.monotonic()
            with self.lock:
                if session is None:
                    session = self.packet['session'] if self.packet else None
            native = CuarmUdp.pack_remote_panel_command(self.codec, normalize_hold(command))
            decode_command(native)
            packed = time.monotonic()
            self._refresh_feedback()
            refreshed = time.monotonic()
            with self.lock:
                self._check_dispatch(now=refreshed, started=started, deadline=deadline,
                                     session=session, phase='before_send')
                self.sequence += 1
                packet = dict(session=session, sequence=self.sequence, echo=self.packet['echo'], command=native,
                              transit=self.transit_sample(started, refreshed))
                rx_stamp = self.rx_stamp
            raw = encode(packet)
            now = time.monotonic()
            with self.lock:
                self._check_dispatch(now=now, started=started, deadline=deadline,
                                     session=session, phase='after_serialization', check_rx=True)
                if rx_stamp is None or not 0 <= now-rx_stamp <= ECHO_AGE_S:
                    raise RuntimeError('V3.1 original command echo expired during serialization; nothing sent')
                traced_send(getattr(self, 'packet_trace', None), self.sock, raw,
                            self.remote, packet, prepared_at=started,
                            producer_deadline_monotonic_s=deadline,
                            feedback_age_ms=(now-rx_stamp)*1000)
                self.timing.update(sent=self.timing['sent']+1, pack_ms=(packed-started)*1000,
                                   max_pack_ms=max(self.timing['max_pack_ms'], (packed-started)*1000),
                                   command_sequence=packet['sequence'],
                                   feedback_sequence=self.packet['sequence'],
                                   session_tag=hashlib.sha256(session.encode()).hexdigest()[:12],
                                   feedback_age_ms=(now-rx_stamp)*1000, dispatch_ms=(now-started)*1000)
                if self.last_send is not None:
                    self.timing['max_send_gap_ms'] = max(self.timing['max_send_gap_ms'], (now-self.last_send)*1000)
                self.last_send = now

    def terminate(self):
        self._closed = True
        self.sock.close()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        trace = getattr(self, 'packet_trace', None)
        if trace is not None:
            trace.close()


class RemoteExecutionGuard(ExecutionGuard):
    def __init__(self, transport, **kwargs):
        self.transport = transport
        super().__init__(self._send_remote, **kwargs)

    def _send_remote(self, command):
        if self.transport.feedback_wait is not None:
            if self.fault:
                self.transport.hold_feedback_wait()
            else:
                self.transport.resume_feedback_wait(command)
        else:
            self.transport.send(command)


class RemoteDirectVisualServo(DirectVisualServo):
    def __init__(self, planner, collision, settings=None):
        super().__init__(planner, collision, settings)
        self.transport = None
        self.waiting_feedback = None
        self.control_context = None
        self.last_receipt = None

    def reset(self):
        super().reset()
        self.waiting_feedback = None
        self.control_context = None
        self.last_receipt = None

    def _target_deadline(self, receipt):
        now = time.monotonic()
        if (type(receipt) not in (int, float) or not math.isfinite(receipt)
                or not 0 <= now-receipt):
            raise ValueError('Direct servo missing command timestamp')
        lease = (self.c['stream_lease_s'] if self.p.target_description == 'book_v3_1_stream'
                 else self.c['probe_timeout_s'])
        return receipt+lease

    def _wait_for_feedback(self, receipt, reason):
        from cuarm_state import CuroboStatus
        guard = self.p._v31_bridge.guard
        guard.update(None)
        if guard.fault:
            raise ValueError(guard.fault)
        self.output_ready = False
        self.paused = True
        self.p.status = CuroboStatus.PLANNING
        self.p._v31_bridge.checked_start = None
        guard.published_key = guard.published_request = None
        first = self.waiting_feedback is None
        self.waiting_feedback = self.transport.begin_feedback_wait(self._target_deadline(receipt))
        if first:
            remaining = self.waiting_feedback['deadline']-time.monotonic()
            print(f'V3.1 direct feedback wait: {reason}; remaining_ms={remaining*1000:.1f}; '
                  'fixed hold, no lease renewal', flush=True)

    def compute(self, receipt):
        if self.transport is None:
            raise ValueError('Remote direct servo has no bounded transport')
        guard = self.p._v31_bridge.guard
        if guard is None:
            raise ValueError('Direct servo has no output guard')
        self.output_ready = False
        self.last_receipt = receipt
        with guard.lock:
            if guard.fault:
                raise ValueError(guard.fault)
            context = (self.transport.session, self.transport.epoch)
            if self.control_context is None:
                self.control_context = context
            elif context != self.control_context:
                raise ValueError('Direct servo control session changed; discard previous target')
            try:
                if self.waiting_feedback is not None:
                    self.waiting_feedback = self.transport.check_feedback_wait()
                    if time.monotonic() >= self._target_deadline(receipt):
                        raise ValueError('Direct servo target expired during feedback wait')
                    snapshot = self.p._v31_bridge.snapshot(max_age=.1)
                    if snapshot['panel']._seldon_session != self.control_context[0]:
                        raise ValueError('Direct servo feedback session changed during wait')
                    guard.update(snapshot)
                    if guard.fault:
                        raise ValueError(guard.fault)
                    self.stepper.reset()
                    self.last_tick = time.monotonic()
                super().compute(receipt)
            except SnapshotUnavailable as error:
                self._wait_for_feedback(receipt, str(error))

    def dispatched(self, accepted):
        super().dispatched(accepted)
        guard = self.p._v31_bridge.guard
        with guard.lock:
            if accepted:
                if self.waiting_feedback is not None:
                    if self.transport.feedback_wait is not None:
                        raise ValueError('Direct servo recovery was not committed')
                    elapsed = time.monotonic()-self.waiting_feedback['started']
                    self.waiting_feedback = None
                    print(f'V3.1 direct feedback recovered: wait_ms={elapsed*1000:.1f}; '
                          'fresh checked dispatch, integrator reset', flush=True)
            elif not guard.fault:
                self._wait_for_feedback(self.last_receipt, 'handoff discarded')


class RemotePanelTransport(PanelTransport):
    # Startup receives state before it owns any command stream. A temporary
    # gap here is a silent wait, not permission to resume an active stream.
    WAIT_BEFORE_FIRST_PUBLICATION = True

    def __init__(self, panel, template, stop_state, hold_mode, **kwargs):
        self.remote_panel = panel
        self.session = None
        self.feedback_wait = None
        super().__init__(self._send_remote, panel.receive_with_timestamp,
                         template, stop_state, hold_mode, **kwargs)

    def _send_remote(self, command):
        deadline = self.deadline
        if self.feedback_wait is not None:
            deadline = min(deadline, self.feedback_wait['deadline'])
            if self.clock() >= deadline:
                self._fail('feedback wait deadline expired before dispatch')
                self._check()
        try:
            self.remote_panel.send_strict(command, deadline=deadline, session=self.session)
        except Exception as error:
            if self.feedback_wait is not None:
                self._fail(f'feedback wait dispatch failed: {error}')
            raise

    def _panel(self):
        try:
            panel = super()._panel()
        except Exception as error:
            if self.feedback_wait is not None:
                self._fail(f'feedback wait state unavailable: {error}')
            raise
        if panel is None:
            if self.feedback_wait is not None:
                self._fail('feedback wait interrupted by STOP or invalid feedback')
            if self.mode == 'idle':
                self.last_sent = None
            return None
        session = panel._seldon_session
        if self.feedback_wait is not None:
            wait = self.feedback_wait
            if session != wait['session'] or self.epoch != wait['epoch']:
                self._fail('feedback wait session changed; explicit control recovery required')
                return None
            if self.clock() >= wait['deadline']:
                self._fail('feedback wait deadline expired; watchdog handoff')
                return None
        if self.session is not None and session != self.session:
            self.epoch += 1
            self.command = None
            self.planning_joints = None
            self.deadline = 0.
            self.mode = 'idle'
            self.last_sent = None
            self._clear_recovery()
        self.session = session
        return panel

    def _fail(self, reason, panel=None):
        if self.feedback_wait is None:
            return super()._fail(reason, panel)
        if self.fault:
            return
        self.fault = 'V3.1 panel transport: ' + reason
        self.command = None
        self.planning_joints = None
        self.running = False
        self.mode = 'fault'
        self._clear_recovery()
        self.log(self.fault)

    def begin_feedback_wait(self, target_deadline):
        from types import MappingProxyType

        with self.lock:
            if not math.isfinite(target_deadline):
                self._fail('invalid feedback wait target deadline')
                self._check()
            if self.feedback_wait is not None:
                self.check_feedback_wait()
                if target_deadline < self.feedback_wait['deadline']:
                    self.feedback_wait = MappingProxyType(dict(
                        self.feedback_wait, deadline=target_deadline))
                    self.deadline = target_deadline
                return self.check_feedback_wait()
            panel = self._eligible()
            now = self.clock()
            deadline = min(now+self.PRODUCER_LEASE,
                           self.panel_stamp+self.FEEDBACK_AGE, target_deadline)
            if self.command is not None:
                deadline = min(deadline, self.deadline)
            self.feedback_wait = MappingProxyType(dict(
                started=now, deadline=deadline, epoch=self.epoch,
                session=self.session, stamp=self.panel_stamp))
            try:
                self.command = self._hold(panel)
                self.planning_joints = None
                self.mode = 'feedback_hold'
                self.deadline = deadline
                self.check_feedback_wait()
                self._emit()
            except Exception as error:
                self._fail(f'feedback wait could not begin: {error}')
                raise
            return self.feedback_wait

    def check_feedback_wait(self):
        with self.lock:
            self._check()
            if self.feedback_wait is None:
                raise RuntimeError('V3.1 feedback wait is not active')
            panel = self._panel()
            self._check()
            if panel is None or self.cycle_epoch != self.epoch:
                self._fail('feedback wait control epoch changed')
                self._check()
            return self.feedback_wait

    def hold_feedback_wait(self):
        with self.lock:
            wait = self.check_feedback_wait()
            self._emit()
            return wait

    def resume_feedback_wait(self, command):
        with self.lock:
            try:
                self.check_feedback_wait()
                super().send(command)
            except Exception as error:
                self._fail(f'feedback wait resume rejected: {error}')
                raise
            self.feedback_wait = None

    def send(self, command):
        with self.lock:
            if self.feedback_wait is not None:
                self.check_feedback_wait()
                raise RuntimeError('V3.1 feedback wait requires checked resume before motion')
            return super().send(command)

    def begin_planning(self, start_joints=None):
        with self.lock:
            if self.feedback_wait is not None:
                self.check_feedback_wait()
                raise RuntimeError('V3.1 feedback wait cannot be replaced by planning hold')
            return super().begin_planning(start_joints)

    def idle_hold(self, cancel_recovery=False):
        with self.lock:
            if self.feedback_wait is not None:
                return self.hold_feedback_wait()
            return super().idle_hold(cancel_recovery)

    def tick(self):
        with self.lock:
            if self.feedback_wait is None:
                return super().tick()
            if self.fault or self._stop.is_set():
                return
            self.hold_feedback_wait()

    def terminate(self):
        with self.lock:
            waiting = self.feedback_wait is not None
            if waiting:
                if not self.fault and not self._stop.is_set() and self.command is not None:
                    try:
                        self.hold_feedback_wait()
                    except Exception as error:
                        self.log(f'V3.1 shutdown feedback hold failed: {error}')
                self.command = None
            self._stop.set()
        if not waiting:
            return super().terminate()
        if self._thread is not None:
            self._thread.join(timeout=1.)


def install_remote_path():
    check_profile()
    import numpy as np
    import curobo_planner_v3_1
    from v3_1_panel_path import PanelPath

    class RemotePath(PanelPath):
        def __init__(self):
            reference = PROFILE/'rt_reference'
            source = reference/'cuarm_rt_control/cuarm_robotics/src'
            wrapper = ROOT/'scripts/v3_1_panel_path_bridge.cpp'
            files = [wrapper, source/'interpolation.cpp', source/'interpolation.h',
                     reference/'robot_platform_utils/cpp/include/eigen_vector.h',
                     reference/'robot_platform_utils/cpp/include/cuarm_state.h']
            digest = hashlib.sha256(b''.join(path.read_bytes() for path in files)).hexdigest()[:20]
            cache = ROOT/'output/v31_seldon_path'
            cache.mkdir(parents=True, exist_ok=True)
            library = cache/f'path_{digest}.so'
            if not library.exists():
                temporary = library.with_suffix(f'.{os.getpid()}.tmp')
                config = Path(os.environ.get('CUARM_CONFIG_ROOT', str(ROOT.parents[1]/'dual_v2_2_configuration')))
                subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                                '-I/usr/include/eigen3', '-I'+str(reference), '-I'+str(source), '-I'+str(config),
                                '-I'+str(config/'robot_platform_utils/cpp/include'),
                                str(wrapper), str(source/'interpolation.cpp'), '-o', str(temporary)],
                               check=True, capture_output=True, text=True, timeout=60)
                temporary.replace(library)
            self.library = ctypes.CDLL(str(library))
            pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            self.library.v31_sample_path.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer,
                ctypes.c_double, pointer, ctypes.c_double, ctypes.c_int, pointer, pointer]
            self.library.v31_sample_path.restype = ctypes.c_int
            self.library.v31_path_error.restype = ctypes.c_char_p

    curobo_planner_v3_1.PanelPath = RemotePath
    return RemotePath


def configure_planner(planner, panel):
    import numpy as np
    limits = np.asarray(panel.packet['speed_limits'], dtype=float)
    planner._v31_bridge.joint_velocity_limits = limits.copy()
    planner._v31_bridge.limits = limits*.1
    planner._v31_direct.c['raw_velocity_ratio'] = min(.3, math.radians(30)/float(limits.max()))
    original = planner._v31_direct
    planner._v31_direct = RemoteDirectVisualServo(planner, original.collision, original.c)
    print(f'155 V3.1: native RAW, remote interpolation sampler, {ECHO_AGE_S*1000:g}ms echo, 250ms lease; '
          f'RAW ratio={planner._v31_direct.c["raw_velocity_ratio"]:.4f}')
