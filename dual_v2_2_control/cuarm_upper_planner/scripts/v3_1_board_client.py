"""Upper-side Cartesian proxy. Differential joint control runs on the robot."""
import copy
import math
import tempfile
import time
import numpy as np
import v3_1_seldon as legacy
from v3_1_seldon import RemotePanel,RemotePanelTransport,RemoteExecutionGuard
from v3_1_board_export import export
from v3_1_panel_transport import ControlSessionCancelled


class BoardPanel(RemotePanel):
    TRACE_PACKETS = True
    RECOVER_SESSION_CANCELLATION = True

    def _consume_batch(self, batch):
        super()._consume_batch(batch)
        fault = (self.packet or {}).get('board_servo', {}).get('fault_details')
        if isinstance(fault, dict):
            key = (fault.get('time_monotonic'), fault.get('reason'))
            if key != getattr(self, '_traced_fault', None):
                self._traced_fault = key
                self._trace_fault('board_fault', board_reason=fault.get('reason'),
                                  board_monotonic_s=fault.get('time_monotonic'))

    def send_strict(self,command,deadline=None,session=None):
        trace = getattr(self, 'packet_trace', None)
        if trace is not None:
            trace.record('dispatch_requested', producer_deadline_monotonic_s=deadline,
                         target_id=(getattr(command, '_board_target', None) or {}).get('target_id'))
        try:
            return self._send_board_strict(command, deadline, session)
        except Exception as error:
            self._trace_fault('dispatch_failed', error_type=type(error).__name__,
                              reason_detail=str(error)[:400])
            raise

    def _send_board_strict(self,command,deadline=None,session=None):
        mode=getattr(command.arm_target_mode,'value',command.arm_target_mode)
        payload=getattr(command,'_board_target',None)
        # ExecutionGuard encodes cancellation as a two-point measured hold
        # while retaining RAW mode. Never replay its copied Cartesian payload.
        if mode!=2 or payload is None or command.arm_waypoint_size!=0:
            return super().send_strict(command,deadline,session)
        from cuarm_state import PanelRemoteState
        with self.send_lock:
            started=time.monotonic()
            self._refresh_feedback()
            with self.lock:
                now=time.monotonic()
                current_session=self.packet['session'] if self.packet else None
                self._check_dispatch(now=now, started=started, deadline=deadline,
                    session=session, phase='cartesian_before_send', require_deadline=True)
                if self.packet.get('board_servo_sha')!=payload['board_servo_sha']:
                    raise ValueError('Board servo bundle mismatch')
                self.sequence+=1
                packet=dict(payload,kind='cartesian',session=current_session,
                            sequence=self.sequence,echo=self.packet['echo'],
                            grippers=copy.deepcopy(payload['grippers']),
                            transit=self.transit_sample(started,now))
                raw=legacy.encode(packet)
                sent_at=time.monotonic()
                self._check_dispatch(now=sent_at, started=started, deadline=deadline,
                    session=session, phase='cartesian_after_serialization',
                    check_rx=True, require_deadline=True)
                legacy.traced_send(getattr(self, 'packet_trace', None), self.sock, raw,
                                   self.remote, packet, prepared_at=started,
                                   producer_deadline_monotonic_s=deadline,
                                   feedback_age_ms=(sent_at-self.rx_stamp)*1000)
                self.timing.update(sent=self.timing.get('sent',0)+1,
                                   command_sequence=packet['sequence'],
                                   feedback_sequence=self.packet['sequence'],
                                   feedback_age_ms=(sent_at-self.rx_stamp)*1000,
                                   dispatch_ms=(sent_at-started)*1000)


class BoardTransport(RemotePanelTransport):
    cancellation_pending = None

    def _cancel_cycle(self, reason):
        # No hold is sent to a revoked owner. Keep faults latched, and notify
        # the main loop even when the publisher observed the transition first.
        self.command = self.planning_joints = self.last_sent = None
        self.deadline = 0.
        self.mode = 'idle'
        self._clear_recovery()
        self.cancellation_pending = 'V3.1 panel session changed; discard result: '+reason

    def _panel(self):
        epoch = self.epoch
        panel = super()._panel()
        if self.epoch != epoch and not self.fault:
            self._cancel_cycle('fresh STOP or session transition')
        return panel

    def _eligible(self):
        self._check()
        panel = self._panel()
        self._check()
        if self.cancellation_pending:
            raise ControlSessionCancelled(self.cancellation_pending)
        if panel is None or self.cycle_epoch != self.epoch:
            raise RuntimeError('V3.1 panel unavailable during computation; discard result')
        if self.command is not None and self.clock() > self.deadline:
            self._fail('planner producer/deadline expired; discard late result', panel)
            self._check()
        return panel

    def begin_cycle(self):
        with self.lock:
            panel = super().begin_cycle()
            if self.cancellation_pending:
                reason, self.cancellation_pending = self.cancellation_pending, None
                raise ControlSessionCancelled(reason)
            return panel

    def _send_remote(self, command):
        try:
            return super()._send_remote(command)
        except ControlSessionCancelled as error:
            self.epoch += 1
            self._cancel_cycle(str(error))
            raise

    def _hold(self,panel):
        result=super()._hold(panel)
        if hasattr(result,'_board_target'):del result._board_target
        return result


class BoardProxy:
    atomic_target_receipt = True

    def __init__(self,planner,panel,settings,digest):
        self.p=planner;self.panel=panel;self.c=settings;self.digest=digest;self.transport=None
        self.next_key=0;self.active=False;self.output_ready=False;self.paused=False;self.recovering_z=False
        self.identity=None;self.payload=None
    def reset(self):
        self.active=False;self.output_ready=False;self.identity=None;self.payload=None
        self.paused=False;self.recovering_z=False
    def compute(self,receipt):
        from cuarm_state import CuroboStatus
        now=time.monotonic();p=self.p;b=p._v31_bridge
        self.output_ready=False
        if receipt is None or not 0<=now-receipt<=self.c['probe_timeout_s']:
            raise ValueError('Board proxy target receipt invalid/expired')
        # The board checks its local sample at 100ms; this upper-side check is
        # the existing 250ms communication freshness policy, not a joint tick.
        snapshot=b.snapshot(max_age=.25)
        target=np.c_[p.target_arm_tool_position,p.target_arm_tool_orientation].copy()
        if target.shape!=(2,7) or not np.isfinite(target).all():raise ValueError('Invalid board target')
        if not self.active:
            b.guard.reset();b.guard.begin(float(target[1,2]));self.active=True
        b.guard.update(snapshot)
        if b.guard.fault:raise ValueError(b.guard.fault)
        # A new agent packet may deliberately repeat the same pose after a
        # fresh image. Cached reads and panel heartbeats retain their receipt;
        # they must not renew the visual target lease.
        p.set_gripper_command()
        grippers=copy.deepcopy(p.curobo_cmd.gripper_joint_command)
        identity=(receipt,p.target_description,target.tobytes(),tuple(tuple(row) for row in grippers))
        with self.panel.lock:
            packet=copy.deepcopy(self.panel.packet)
        if packet.get('board_servo_sha')!=self.digest:raise ValueError('Board servo contract changed')
        status=packet.get('board_servo',{})
        if status.get('fault'):raise ValueError(status['fault'])
        if identity!=self.identity:
            if now-receipt>self.c['stream_lease_s']:raise ValueError('New board visual target is stale')
            target_echo=packet['echo']-(now-receipt)
            if self.identity is not None and receipt==self.identity[0]:
                # A revision of one cached receipt cannot refresh its lease.
                target_echo=self.payload['target_echo']
            self.next_key+=1;self.identity=identity
            # Anchor expiry to robot monotonic time from the echoed feedback.
            # Repeated publication reuses this anchor and target_id unchanged.
            self.payload=dict(board_servo_sha=self.digest,target_id=self.next_key,
                action=p.target_description,target_echo=target_echo,pose=target.tolist(),grippers=grippers)
        self.paused=bool(status.get('paused'));self.recovering_z=bool(status.get('recovering_z'))
        b.serial=self.next_key;b.checked_start=snapshot;b.requested_target=target
        p.curobo_cmd.arm_joint_command=snapshot['joints'].reshape(2,7).tolist()
        p.curobo_cmd.waypoints_size=0;p.curobo_cmd.waypoints=[];p.curobo_cmd.cmd_plan_size=0
        p.status=CuroboStatus.PLANNING_SUCCEEDED;self.output_ready=True
    def configure_panel(self,command):
        command.interpolation_acceleration_time=0.;command.no_interpolation_max_velocity_ratio=.3
        # The ordinary converter reuses the panel object and leaves WAYPOINT
        # fields behind when switching to RAW. Clear them for a new Cartesian
        # target; ExecutionGuard may later add its two-point cancellation hold.
        command.arm_waypoint_size=0
        command.arm_waypoint_dt=0.
        command.arm_waypoint_command=[]
        command._board_target=copy.deepcopy(self.payload)
        command.gripper_joint_command=copy.deepcopy(self.payload['grippers'])
    def dispatched(self,accepted):pass  # ACK is read from robot, never assumed here.


class BoardGuard(RemoteExecutionGuard):
    def publish(self,command,checked_start,key,requested=None):
        if getattr(command,'_board_target',None) is None:return super().publish(command,checked_start,key,requested)
        with self.lock:
            self.update(None)
            if self.fault:return False
            self.send(command)
            self.last_command=copy.deepcopy(command)
            self.published_key=key;self.published_request=copy.deepcopy(requested)
            return True
    def acknowledgement(self):
        with self.lock:
            if self.fault:return None,None,self.fault
            with self.transport.remote_panel.lock:
                packet=copy.deepcopy(self.transport.remote_panel.packet)
            status=(packet or {}).get('board_servo',{})
            if status.get('fault'):return None,None,str(status['fault'])
            if not status.get('active') or status.get('target_id')!=self.published_key:
                return None,None,None
            return self.published_key,copy.deepcopy(self.published_request),None


def configure(planner,panel):
    board_digest = (panel.packet or {}).get('board_servo_sha')
    if not board_digest:
        raise ValueError(
            'Board servo backend unavailable: feedback has no board_servo_sha. '
            'dual_v2_2_upper_planner_board.py requires robot backend '
            'seldon_arm.server_board, not seldon_arm.server_v31. '
            'Use matching backend/planner entries; do not run both backends together')
    # Export again from the active solver config; byte-identical identity is
    # required. No trusting a separately tuned URDF or copying joint zeros.
    with tempfile.TemporaryDirectory(prefix='board-model-check-') as directory:
        digest=export(planner,directory)
    if board_digest!=digest:
        raise ValueError(f'Board model/code differs: local={digest}, board={board_digest}; '
                         'export and deploy the matching board bundle first')
    limits=np.asarray(panel.packet['speed_limits'],float)
    planner._v31_bridge.joint_velocity_limits=limits.copy()
    planner._v31_bridge.limits=limits*.1
    from book_planning_budget import install as install_book_planning_budget
    install_book_planning_budget(planner)
    planner._v31_direct=BoardProxy(planner,panel,copy.deepcopy(planner._v31_direct.c),digest)
    print('Visual backend: ROBOT-LOCAL differential servo; upper sends Cartesian targets only; board ACK required',flush=True)


def install():
    from v3_1_panel_process import ProcessBoardPanel, ProcessBoardTransport
    legacy.RemotePanel=ProcessBoardPanel
    legacy.RemotePanelTransport=ProcessBoardTransport
    legacy.RemoteExecutionGuard=BoardGuard
    legacy.configure_planner=configure
