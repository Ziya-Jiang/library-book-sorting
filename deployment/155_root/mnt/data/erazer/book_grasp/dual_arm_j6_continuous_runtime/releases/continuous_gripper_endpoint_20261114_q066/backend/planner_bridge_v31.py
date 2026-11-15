"""Opt-in V3.1 bridge. The original cockpit bridge is left unchanged."""
from .gripper_policy import validate_gripper_targets
import hashlib
import json
import math
from pathlib import Path
import secrets
import time

from .planner_bridge import PlannerBridge as LegacyBridge, encode_state
from .v31_wire import CommandGate, ECHO_AGE_S, LEASE_S, decode, encode
from .v31_timing import TimedPlannerBridge


class PlannerBridge(TimedPlannerBridge, LegacyBridge):
    def __init__(self, controller):
        super().__init__(controller)
        self.mode = 0
        self.ratio = .1
        self.session = secrets.token_hex(16)
        self.gate = CommandGate(self.session)
        self.state_sequence = 0
        self.command_seen = False
        manifest = json.loads(Path(__file__).with_name('v31_contract.json').read_text())
        self.contract = {}
        for name, item in manifest.items():
            actual = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
            if actual != item['sha256']:
                raise ValueError('V3.1 execution contract changed: '+name)
            self.contract[name] = actual

    def disarm(self):
        super().disarm()
        self.mode, self.ratio = 0, .1
        self.session = secrets.token_hex(16)
        self.gate = CommandGate(self.session)
        self.command_seen = False
        self._reset_timing_epoch()
        self.status = 'V3.1 控制已撤销；需重新允许'

    def arm(self):
        super().arm()
        self.session = secrets.token_hex(16)
        self.gate = CommandGate(self.session)
        self.command_seen = False
        self.last_fault = None
        self._reset_timing_epoch()
        self.status = 'V3.1 已允许；等待首条有效指令，收到后启用 250ms 断流保护'

    def stop_fault(self, reason):
        self.last_fault = dict(reason=reason, time_monotonic=time.monotonic(),
                               command_stream=self.gate.diagnostics(),
                               transport_timing=self.timing_snapshot(peek=True))
        self.c.stop(reason)
        print('V3.1 timing at fault: '+json.dumps(self.last_fault), flush=True)

    def validate_targets(self, command):
        controller = self.c
        validate_gripper_targets(controller, command)
        points = command['points'] or [command['target']]
        converted = []
        for point in points:
            row = []
            for arm_index in range(2):
                arm = controller.state['arms'][arm_index]
                values = []
                for joint in range(7):
                    soft = controller.calibration.bounds(arm_index, joint, arm['soft_position'][joint])
                    hard = controller.calibration.bounds(arm_index, joint, arm['hard_position'][joint])
                    lower, upper = max(soft[0], hard[0]), min(soft[1], hard[1])
                    if controller.model_limits:
                        model = controller.model_limits[arm_index][joint]
                        lower, upper = max(lower, model[0]), min(upper, model[1])
                    if not lower <= point[arm_index][joint] <= upper:
                        raise ValueError(f'V3.1 arm {arm_index} joint {joint+1} outside limits')
                    raw = controller.calibration.to_rt(arm_index, joint, point[arm_index][joint])
                    delta = abs(raw-arm['position'][joint])
                    if command['mode'] == 2 and command['ratio']*arm['soft_velocity'][joint][1] > math.radians(30)+1e-9:
                        raise ValueError('V3.1 RAW speed cap exceeds 30 deg/s')
                    if command['mode'] == 2 and delta > .03:
                        raise ValueError('V3.1 RAW measured lead exceeds 0.03 rad')
                    if command['mode'] == 0:
                        speed = min(math.radians(30), arm['soft_velocity'][joint][1],
                                    -arm['soft_velocity'][joint][0], arm['hard_velocity'][joint][1],
                                    -arm['hard_velocity'][joint][0])
                        if math.pi*.5*delta/command['duration'] > speed:
                            raise ValueError('V3.1 single point speed exceeds limit')
                    values.append(raw)
                row.append(values)
            converted.append(row)
        for values, gripper in zip(command['grippers'], controller.state['grippers']):
            for value, soft, hard in zip(values, gripper['soft_position'], gripper['hard_position']):
                if not max(0, soft[0], hard[0]) <= value <= min(1, soft[1], hard[1]):
                    raise ValueError('V3.1 gripper outside limits')
        return converted

    def tick(self):
        if not self.sock:
            return
        now = time.monotonic()
        self._begin_timed_tick(now)
        if self.enabled and (not self.c.active or not self.c.fresh()):
            self.stop_fault('V3.1 feedback/control lost')
        for attempt in range(32):
            try:
                raw, sender = self._receive_command()
            except BlockingIOError:
                break
            except ValueError as error:
                self.stop_fault('V3.1 command receive timing invalid: '+str(error))
                break
            if sender != (self.settings['ip'], self.settings['port']):
                self.rejected += 1
                continue
            try:
                packet = decode(raw)
                self._timing().packet(packet, "decoded")
                if not self.enabled:
                    self._timing().packet(packet, "disabled")
                    continue
                self._check_timing_deadlines(time.monotonic())
                if not self.enabled:
                    break
                self._timing().check_fresh(time.monotonic())
                command = self.gate.accept(packet, time.monotonic())
                if command is None:
                    self._timing().packet(packet, "obsolete")
                    self._timing().current['discard_reason'] = self.gate.last_discard['reason']
                    continue
                self.c.check_live()
                if not all(all(arm['enabled']) for arm in self.c.state['arms']):
                    raise ValueError('V3.1 requires all joints enabled')
                identity = dict(command)
                if command['points']:
                    identity['target'] = []
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
                if digest != self.last_digest:
                    converted = self.validate_targets(command)
                    self.c.target = converted[-1]
                    self.c.duration = command['duration']
                    self.c.gripper_target = command['grippers']
                    self.waypoints = converted if command['points'] else []
                    self.waypoint_dt = command['dt']
                    self.mode, self.ratio = command['mode'], command['ratio']
                    self.last_digest = digest
                self._accept_timed_command()
                self.last_rx = now
                self.command_seen = True
                self.rx_count += 1
                self.status = f'V3.1 active mode={self.mode}; lease 250ms'
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.rejected += 1
                self.stop_fault('V3.1 rejected command: '+str(error))
        if now-self.last_tx >= .01 and self.c.fresh() and self.c._gripper_protocol_ok():
            self.state_sequence += 1
            speed_limits = [[min(pair[1], -pair[0]) for pair in arm['soft_velocity']]
                            for arm in self.c.state['arms']]
            packet = dict(kind='state', session=self.session, sequence=self.state_sequence,
                          echo=now, rt_timestamp=self.c.state['timestamp'],
                          state=encode_state(self.c, self.enabled).decode('ascii'),
                          command_receipt=self.command_receipt(),
                          contract=self.contract, speed_limits=speed_limits, echo_age_s=ECHO_AGE_S)
            self.sock.sendto(encode(packet), (self.settings['ip'], self.settings['port']))
            self.last_tx = now
            self.tx_count += 1

    def snapshot(self):
        return dict(super().snapshot(), protocol='seldon-v31-1', mode=self.mode,
                    command_lease_s=LEASE_S, echo_age_s=ECHO_AGE_S,
                    command_stream=self.gate.diagnostics(),
                    transport_timing=self.timing_snapshot(),
                    fault_details=getattr(self, "last_fault", None),
                    waiting_first_command=self.enabled and not self.command_seen)
