"""Independent full visual-loop bridge; original server_board is unchanged."""
from .active_arm_policy import required_joints_enabled, validate_static_left
import hashlib
import json
import time
from .planner_bridge_board import PlannerBridge as BoardBridge
from .planner_bridge import encode_state
from .v31_wire import LEASE_S, ECHO_AGE_S, decode, encode
from .board_visual_ipc import VisualIPC
from .packet_trace import PacketTrace, packet_identity, traced_send

class PlannerBridge(BoardBridge):
    def __init__(self,controller):
        super().__init__(controller)
        self.visual=None
        self.packet_trace=PacketTrace(self.c.root/'runtime/packet_trace', 'board')
        print('155逐包日志：'+str(self.packet_trace.directory), flush=True)

    def timing_snapshot(self, peek=False):
        result=super().timing_snapshot(peek)
        trace=getattr(self,'packet_trace',None)
        if trace is not None:result['packet_trace']=trace.status()
        return result

    def stop_fault(self,reason):
        trace=getattr(self,'packet_trace',None)
        if trace is not None:
            trace.fault(reason, accepted_sequence=self.gate.sequence,
                        session_tag=hashlib.sha256(self.session.encode()).hexdigest()[:12],
                        timing=self._timing().snapshot(time.monotonic()),
                        rt_feedback_age_ms=(time.monotonic()-self.c.received)*1000
                        if getattr(self.c,'received',None) else None)
        return super().stop_fault(reason)

    def _receive_command(self):
        trace=getattr(self,'packet_trace',None)
        try:raw,sender=super()._receive_command()
        except ValueError as error:
            if trace is not None:
                trace.record('receive_error', error_type=type(error).__name__)
            raise
        if trace is not None:
            trace.record('receive', wire_sha256=hashlib.sha256(raw).hexdigest(),
                         size_bytes=len(raw), **dict(self._timing().current))
        return raw,sender

    def disarm(self):
        trace=getattr(self,'packet_trace',None)
        if trace is not None:trace.record('control_disarmed')
        if getattr(self,'visual',None):
            self.visual.cancel('control revoked',detail=getattr(self.c,'_stop_context',None))
        return super().disarm()

    def disconnect(self):
        trace=getattr(self,'packet_trace',None)
        if trace is not None:trace.record('planner_disconnected')
        if getattr(self,'visual',None):
            self.visual.close();self.visual=None
        return super().disconnect()

    def servo_status(self):
        result=super().servo_status()
        handoff=getattr(getattr(self,'visual',None),'handoff',None)
        if handoff is not None and handoff.accepted is not None:
            result.update(handoff.status())
        return result

    def tick(self):
        if not self.sock:
            return
        trace=getattr(self,'packet_trace',None)
        started=time.monotonic()
        if trace is not None:
            loop=getattr(self.c,'loop_timing',None)
            previous=dict(loop.recent[-1]) if loop is not None and loop.recent else None
            trace.record('control_tick_begin', enabled=self.enabled, previous_cycle=previous,
                         rt_feedback_age_ms=(started-self.c.received)*1000
                         if getattr(self.c,'received',None) else None)
        try:return self._tick_with_trace()
        finally:
            if trace is not None:
                trace.record('control_tick_return', enabled=self.enabled,
                             tick_ms=(time.monotonic()-started)*1000)

    def _tick_with_trace(self):
        now = time.monotonic()
        self._begin_timed_tick(now)
        if self.enabled and (not self.c.active or not self.c.fresh()):
            self.stop_fault('V3.1 feedback/control lost')
        if self.visual is None:self.visual=VisualIPC(self,self.c.root/'runtime/board_visual.json')
        self.visual.poll()
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
            packet=None
            packet_error=None
            received_timing=self._timing().current
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
                if self.visual.external(command,grippers=packet.get('grippers')):
                    self._accept_timed_command()
                    self.last_rx=now;self.command_seen=True;self.rx_count+=1
                    continue
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
                packet_error=type(error).__name__+': '+str(error)
                self.rejected += 1
                self.stop_fault('V3.1 rejected command: '+str(error))
            finally:
                trace=getattr(self,'packet_trace',None)
                if trace is not None:
                    identity=packet_identity(packet,raw)
                    identity.update(dict(received_timing or {}))
                    trace.record('command_processed', **identity, error=packet_error,
                                 completed_monotonic_s=time.monotonic())
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
            traced_send(getattr(self,'packet_trace',None), self.sock, encode(packet),
                        (self.settings['ip'],self.settings['port']), packet,
                        prepared_at=now, feedback_echo_monotonic_s=now)
            self.last_tx = now
            self.tx_count += 1
