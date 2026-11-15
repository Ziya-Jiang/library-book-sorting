"""Local, leased visual ownership. Called only by the existing bridge thread."""
from .active_arm_policy import required_joints_enabled, validate_static_left
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import time
from collections import OrderedDict
import numpy as np
from .stop_diagnostics import capture as capture_stop


class VisualSessionExpired(ValueError):
    def __init__(self,detail=None):
        self.stop_detail=detail
        message='Visual owner/session expired'
        if detail:
            message+=(f'; first_stop={detail["reason"]}; source={detail["source"]}; '
                      f'utc={detail["utc"]}; rt_feedback_age_ms={detail["rt_feedback_age_ms"]}; '
                      f'stop_id={detail["id"]}')
        super().__init__(message)


def command_digest(command):
    identity=dict(command)
    if command.get('points'):identity['target']=[]
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).digest()


class VisualIPC:
    lease_s=.5
    def __init__(self,bridge,path):
        self.b=bridge;self.path=Path(path);self.owner=None;self.last_rx=0.;self.saved_engine=None
        self.sequence=0;self.baseline=None;self.blocked_digest=None;self.handoff=None
        self.token=secrets.token_hex(32)
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        self.socket_path=f'/tmp/seldon-visual-{os.getuid()}-{os.getpid()}-{secrets.token_hex(4)}.sock'
        try:
            self.sock.bind(self.socket_path);os.chmod(self.socket_path,0o600);self.sock.setblocking(False)
        except BaseException:
            self.sock.close();raise
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:json.dump(dict(socket=self.socket_path,token=self.token,pid=os.getpid(),protocol=1),f)

    def snapshot(self):
        b=self.b;c=b.c;c.check_live()
        if not b.enabled or not c.active:raise ValueError('Board visual requires armed planner and active control')
        if not required_joints_enabled(c):raise ValueError('Board visual joint disabled')
        now=time.monotonic()
        if not 0<=now-c.received<=.1:raise ValueError('Board visual local feedback stale')
        q=np.asarray(c.calibration.positions([a['position'] for a in c.state['arms']])).reshape(14)
        p,r=b.model.tcp(q)
        velocity=np.asarray([[c.calibration.gain[a][j]*v for j,v in enumerate(arm['velocity'])]
                             for a,arm in enumerate(c.state['arms'])]).reshape(14)
        return dict(position=p[0].tolist(),orientation=r[0].tolist(),joints=q.tolist(),
                    velocity=velocity.tolist(),stamp=c.received,fk_completed=time.monotonic(),
                    accepted=b.engine.accepted,session=b.session)

    def hold(self):
        b=self.b;c=b.c
        b.engine.cancel()
        # Raw measured hold remains in the original owner/RT path. Never open grippers.
        c.target=[list(a['position']) for a in c.state['arms']]
        c.duration=0.;b.mode=2;b.ratio=min(b.ratio,.1);b.waypoints=[];b.waypoint_dt=0.

    def cancel(self,reason,detail=None):
        if reason!='worker completed':self.handoff=None
        if self.owner is not None:
            detail=detail or capture_stop(self.b.c,reason,'visual.cancel')
            if not hasattr(self,'revoked'):self.revoked=OrderedDict()
            self.revoked[self.owner]=detail
            while len(self.revoked)>32:self.revoked.popitem(last=False)
            journal=getattr(self.b.c,'stop_journal',None)
            if journal is not None and detail is not getattr(self.b.c,'_stop_context',None):
                journal.record(detail)
            self.blocked_digest=self.baseline
            self.owner=None
            self.hold()
            if self.saved_engine is not None:
                self.b.engine=self.saved_engine;self.saved_engine=None
                self.b.engine.cancel()
            print('Board visual ownership released: '+reason,flush=True)

    def external(self,command,grippers=None):
        if 'target_id' in command:
            handoff=getattr(self,'handoff',None)
            if handoff is not None:
                if grippers is not None:
                    from .visual_handoff import validated_grippers
                    command=dict(command,grippers=validated_grippers(grippers))
                accepted=handoff.accept(command,self.b.session,self.snapshot(),time.monotonic())
                self.b.engine.last_target_id=handoff.accepted
                self.b.engine.accepted=handoff.accepted
                if grippers is not None:
                    self.b.c.gripper_target=command['grippers']
                return accepted
            if self.owner:self.cancel('external Cartesian command')
            self.blocked_digest=None
            return False
        digest=command_digest(command)
        handoff=getattr(self,'handoff',None)
        if handoff is not None and digest!=self.blocked_digest:
            self.handoff=None  # A subsequent normal action ends the acknowledged handoff.
        if self.owner:
            if digest==self.baseline:return True
            self.cancel('external command changed')
        if digest==self.blocked_digest:return True
        self.blocked_digest=None
        return False

    def handle(self,req,peer):
        if not secrets.compare_digest(str(req.get('token','')),self.token):raise ValueError('Invalid local token')
        op=req.get('op');now=time.monotonic();b=self.b
        if op=='info':
            from .board_servo.baseline_j2 import POLICY
            names=('server_board_visual.py','planner_bridge_visual.py','board_visual_ipc.py',
                   'startup_reservation.py','visual_handoff.py')
            return dict(protocol=1,board_digest=b.model.digest,session=b.session,joint_policies=[POLICY],
                bridge_manifest={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
        if op=='begin':
            from .board_servo.baseline_j2 import POLICY
            policy=req.get('joint_policy')
            if policy not in (None,POLICY):raise ValueError('Unknown visual joint policy')
            self.handoff=None
            if self.owner:raise ValueError('Local visual already owned')
            state=self.snapshot()
            if not b.command_seen or now-b.last_rx>.25:raise ValueError('No live upper planner lease')
            if b.last_digest is None or b.engine.request is not None:raise ValueError('Visual takeover requires ordinary held command')
            if np.max(np.abs(state['velocity']))>.02:raise ValueError('Visual takeover requires settled arms')
            from .board_servo.engine import Engine
            engine=Engine(b.model,joint_policy=policy)
            self.saved_engine=b.engine;self.saved_engine.cancel();b.engine=engine
            self.owner=peer;self.sequence=0;self.last_rx=now;self.session=b.session
            if hasattr(self,'revoked'):self.revoked.pop(peer,None)
            self.baseline=b.last_digest;self.blocked_digest=None;self.left=np.r_[state['position'][0],state['orientation'][0]]
            self.hold()
            return dict(state=state,joint_policy=policy)
        if peer!=self.owner or self.session!=b.session:
            raise VisualSessionExpired(getattr(self,'revoked',{}).get(peer))
        seq=req.get('seq')
        if type(seq) is not int or seq<=self.sequence:raise ValueError('Local visual request reordered')
        self.sequence=seq
        state=self.snapshot();self.last_rx=now
        if op=='state':return dict(state=state)
        if op=='target':
            from .board_servo.wire import target
            packet=dict(req['target'],board_servo_sha=b.model.digest)
            command=target(packet,now,.3,b.model.digest)
            pose=np.asarray(command['pose'])
            if not np.allclose(pose[0],self.left,atol=1e-8,rtol=0):raise ValueError('Local visual left arm target changed')
            # Worker timestamps use the same monotonic clock, never a PC offset.
            b.engine.accept(command,now)
            return dict(state=state)
        if op=='hold':
            self.hold();return dict(state=state)
        if op=='end':
            handoff=None
            if req.get('success') is True:
                from .visual_handoff import MeasuredHandoff
                if self.b.engine.accepted<=0 or req.get('target_id')!=self.b.engine.accepted:
                    raise ValueError('Visual completion target has not been acknowledged')
                handoff=MeasuredHandoff(state,req.get('final_pose'),self.b.session,
                    self.saved_engine.last_target_id,now)
            self.cancel('worker '+('completed' if handoff else 'ended/aborted'))
            self.handoff=handoff
            return dict(state=state,held_pose=handoff.pose if handoff else None)
        raise ValueError('Unknown local visual operation')

    def poll(self):
        if self.owner and (not self.b.enabled or self.session!=self.b.session):self.cancel('control revoked')
        if self.owner and time.monotonic()-self.last_rx>self.lease_s:
            self.cancel('worker lease expired')
            self.b.stop_fault('Board visual worker lease expired')
        for _ in range(8):
            try:raw,peer=self.sock.recvfrom(60001)
            except BlockingIOError:break
            try:
                if len(raw)>60000:raise ValueError('Oversized local visual message')
                req=json.loads(raw);result=self.handle(req,peer);reply=dict(ok=True,**result)
            except Exception as error:
                detail=getattr(error,'stop_detail',None)
                if peer==self.owner:
                    self.cancel('worker request rejected: '+str(error))
                    detail=getattr(self,'revoked',{}).get(peer)
                reply=dict(ok=False,error=str(error))
                if detail is not None:reply['stop_detail']=detail
            try:self.sock.sendto(json.dumps(reply,allow_nan=False).encode(),peer)
            except OSError:
                if peer==self.owner:self.cancel('worker disconnected')

    def close(self):
        self.cancel('bridge disconnected');self.sock.close()
        Path(self.socket_path).unlink(missing_ok=True)
        try:
            if json.loads(self.path.read_text()).get('token')==self.token:self.path.unlink()
        except (OSError,ValueError):pass
