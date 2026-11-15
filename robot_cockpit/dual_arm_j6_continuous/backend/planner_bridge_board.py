"""Opt-in V3.1 bridge. The original cockpit bridge is left unchanged."""
from .active_arm_policy import required_joints_enabled, validate_static_left
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
import numpy as np
from .board_servo.model import Model
from .board_servo.engine import Engine
from .board_servo.wire import target as decode_target


class BoardGate(CommandGate):
    def __init__(self,session,digest):
        super().__init__(session);self.digest=digest

    def accept(self,packet,now):
        if packet.get('kind')!='cartesian':return super().accept(packet,now)
        if not self.check_identity(packet,now):return None
        result=decode_target(packet,now,ECHO_AGE_S,self.digest)
        self.commit(packet)
        return result



class PlannerBridge(TimedPlannerBridge, LegacyBridge):
    def __init__(self, controller):
        super().__init__(controller)
        self.model=Model(Path(__file__).with_name('board_servo'))
        for name,digest in self.model.deployment_hashes.items():
            if Path(name).name!=name or hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Board bridge entry differs from exported model: '+name)
        self.engine=Engine(self.model)
        self.mode = 0
        self.ratio = .1
        self.session = secrets.token_hex(16)
        self.gate = BoardGate(self.session,self.model.digest)
        self.state_sequence = 0
        self.command_seen = False
        manifest = json.loads(Path(__file__).with_name('v31_contract.json').read_text())
        self.contract = {}
        for name, item in manifest.items():
            actual = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
            if actual != item['sha256']:
                raise ValueError('V3.1 execution contract changed: '+name)
            self.contract[name] = actual
        if self.model.contract!=self.contract:raise ValueError('Board model contract mismatch')

    def disarm(self):
        super().disarm()
        self.mode, self.ratio = 0, .1
        self.session = secrets.token_hex(16)
        self.gate = BoardGate(self.session,self.model.digest)
        self.command_seen = False
        self.engine=Engine(self.model)
        self._reset_timing_epoch()
        self.status = 'V3.1 控制已撤销；需重新允许'

    def arm(self):
        super().arm()
        self.last_fault=None
        self.session = secrets.token_hex(16)
        self.gate = BoardGate(self.session,self.model.digest)
        self.command_seen = False
        self.engine=Engine(self.model)
        self._reset_timing_epoch()
        self.status = 'V3.1 已允许；等待首条有效指令，收到后启用 250ms 断流保护'

    def stop_fault(self,reason):
        # disarm reconstructs Engine. Retain the reason across that reset so
        # the upper ACK path receives the cause rather than a generic timeout.
        self.last_fault=dict(reason=reason,target_id=self.engine.last_target_id,
                             time_monotonic=time.monotonic(),
                             command_stream=self.gate.diagnostics(),
                             transport_timing=self.timing_snapshot(peek=True))
        detail=getattr(self.engine,'constraint_diagnostics',None)
        if detail is not None:
            self.last_fault['joint_constraints']=detail
        self.c.stop(reason)
        print(reason,flush=True)
        print('V3.1 timing at fault: '+json.dumps(self.last_fault),flush=True)

    def servo_status(self):
        result=self.engine.status()
        fault=getattr(self,'last_fault',None)
        if fault:
            result['fault']=fault['reason'];result['fault_details']=dict(fault)
        return result

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
        validate_static_left(controller, converted)
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
                if not required_joints_enabled(self.c):
                    raise ValueError('V3.1 requires all joints enabled')
                if 'target_id' in command:
                    self.engine.accept(command,now)
                    self._accept_timed_command()
                    self.last_rx=now;self.command_seen=True;self.rx_count+=1
                    self.status='Board Cartesian target received'
                    continue
                self.engine.cancel()
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
        if self.enabled and self.engine.request is not None:
            try:self._servo_tick()
            except Exception as error:
                self.engine.fault=str(error)
                self.stop_fault('Board servo stopped: '+str(error))
        if now-self.last_tx >= .01 and self.c.fresh() and self.c._gripper_protocol_ok():
            self.state_sequence += 1
            speed_limits = [[min(pair[1], -pair[0]) for pair in arm['soft_velocity']]
                            for arm in self.c.state['arms']]
            packet = dict(kind='state', session=self.session, sequence=self.state_sequence,
                          echo=now, rt_timestamp=self.c.state['timestamp'],
                          state=encode_state(self.c, self.enabled).decode('ascii'),
                          command_receipt=self.command_receipt(),
                          contract=self.contract, speed_limits=speed_limits, echo_age_s=ECHO_AGE_S,
                          board_servo_sha=self.model.digest,board_servo=self.servo_status())
            self.sock.sendto(encode(packet), (self.settings['ip'], self.settings['port']))
            self.last_tx = now
            self.tx_count += 1

    def _servo_tick(self):
        c=self.c;c.check_live()
        if not required_joints_enabled(c):raise ValueError('Board joint disabled')
        q=np.asarray(c.calibration.positions([a['position'] for a in c.state['arms']])).reshape(14)
        velocity=np.asarray([[c.calibration.gain[a][j]*v for j,v in enumerate(arm['velocity'])]
                             for a,arm in enumerate(c.state['arms'])]).reshape(14)
        position,orientation=self.model.tcp(q)
        snapshot=dict(joints=q,velocity=velocity,position=position[0],orientation=orientation[0],stamp=c.received)
        bounds=[]
        for a,arm in enumerate(c.state['arms']):
            for j in range(7):
                soft=c.calibration.bounds(a,j,arm['soft_position'][j]);hard=c.calibration.bounds(a,j,arm['hard_position'][j])
                limits=c.model_limits[a][j] if c.model_limits else (-math.inf,math.inf)
                bounds.append([max(soft[0],hard[0],limits[0]),min(soft[1],hard[1],limits[1])])
        raw_grip=np.array([v for g in c.state['grippers'] for v in g['position']])
        grip=self.model.grip_matrix@raw_grip+self.model.grip_bias
        vmax=max(pair[1] for a in c.state['arms'] for pair in a['soft_velocity'])
        ratio=min(.3,math.radians(30)/vmax)
        command=self.engine.tick(snapshot,grip,np.array(bounds).T,ratio)
        if command is None:return
        native=dict(mode=2,duration=0.,ratio=ratio,target=command.reshape(2,7).tolist(),
                    points=[],dt=0.,grippers=c.gripper_target)
        converted=self.validate_targets(native)
        c.target=converted[-1];c.duration=0.;self.waypoints=[];self.waypoint_dt=0.
        self.mode=2;self.ratio=ratio;self.last_digest=None
        self.engine.commit()

    def snapshot(self):
        return dict(super().snapshot(), protocol='seldon-v31-1', mode=self.mode,
                    command_lease_s=LEASE_S, echo_age_s=ECHO_AGE_S,
                    waiting_first_command=self.enabled and not self.command_seen,
                    command_stream=self.gate.diagnostics(),
                    transport_timing=self.timing_snapshot(),
                    board_servo_sha=self.model.digest,board_servo=self.servo_status())
