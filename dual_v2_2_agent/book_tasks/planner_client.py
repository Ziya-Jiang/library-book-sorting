"""Route-2 commands over the existing session-controlled Planner channel.

Local JSON files contain bounded geometry requests and diagnostics, never joint
trajectories. Only the Planner owns expiring, single-use executable tickets.
"""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .shelf import ShelfRejected, transform

PROTOCOL = 1


def ipc_path(ticket, suffix):
    if not re.fullmatch('[a-f0-9]{32}', ticket) or suffix not in ('request', 'report'):
        raise ShelfRejected('Invalid task Planner IPC identity')
    port = int(os.environ.get('CUARM_V31_TARGET_LOCAL_PORT', '10176'))
    root = Path(f'/tmp/book-task-planner-{os.getuid()}-{port}')
    return root / (ticket+'.'+suffix+'.json')


def process_start(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (ValueError, OSError, IndexError) as error:
        raise ShelfRejected('Task Planner process is no longer running') from error


def capability_path():
    return ipc_path('0'*32, 'report').parent/'capability.json'


def require_loaded_planner():
    """Read-only startup gate: no socket, process launch, GPU or robot command."""
    path = capability_path()
    try:
        data = json.loads(path.read_text())
        if (data.get('protocol') != PROTOCOL or data.get('planner_sha') != planner_code_digest()
                or data.get('start_ticks') != process_start(data['pid'])
                or data.get('port') != int(os.environ.get('CUARM_V31_TARGET_LOCAL_PORT', '10176'))
                or data.get('capabilities') != ['shelf_scene_v1', 'attached_right_book', 'rt_curve_check']):
            raise ShelfRejected('Running Planner does not match the installed route-2 code/profile')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ShelfRejected('Shelf mode needs the updated Planner loaded in a maintenance window; no motion enabled') from error
    return data


def planner_code_digest():
    """Fingerprint modules actually loaded by the task Planner, on this PC."""
    root = Path(__file__).resolve().parents[2]
    scripts = root/'dual_v2_2_control/cuarm_upper_planner/scripts'
    paths = [scripts/name for name in ('book_task_planner.py', 'curobo_planner_v3_1.py',
        'book_planning_budget.py', 'book_planning_diagnostics.py',
        'baseline_approach_planner.py', 'v3_1_motion_bridge.py', 'v3_1_panel_path.py')]
    paths += [Path(__file__).with_name(name) for name in
              ('planner_client.py', 'planner_scene.py', 'shelf.py', 'placement.py')]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


class ScenePlannerClient:
    def __init__(self, robot, bundle_sha, config_sha, checkpoint=lambda: None, log=lambda event: None):
        for value in (bundle_sha, config_sha):
            if not re.fullmatch('[a-f0-9]{64}', value):
                raise ShelfRejected('Task bundle/config digest is required')
        self.robot, self.bundle_sha, self.config_sha = robot, bundle_sha, config_sha
        self.checkpoint, self.log = checkpoint, log
        self.scene_version = None
        self.planner_sha = planner_code_digest()
        self.pending = {}

    def _pose(self):
        from baseline_approach_client import pose
        state, q, xyz, angles = pose(self.robot)
        t = np.eye(4); t[:3, 3] = xyz*.001
        t[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        return state, q, t

    def _command(self, description, target):
        from cuarm_state import TargetCommandArmCommandType
        state, _, _ = self._pose()
        command = copy.deepcopy(self.robot.target_command)
        command.arm_command_type = TargetCommandArmCommandType.CARTESIAN
        command.arm_tool_command = np.asarray(command.arm_tool_command, float).copy()
        command.arm_tool_command[0] = state['left_arm']['position'] + state['left_arm']['orientation']
        xyzw = Rotation.from_matrix(target[:3, :3]).as_quat()
        command.arm_tool_command[1] = np.r_[target[:3, 3], xyzw[[3, 0, 1, 2]]]
        command.description = description
        return command

    def _wait(self, ticket, sent, stage, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.checkpoint()
            path = ipc_path(ticket, 'report')
            if path.exists():
                report = json.loads(path.read_text())
                if (report.get('ticket') != ticket or report.get('protocol') != PROTOCOL
                        or report.get('planner_sha') != self.planner_sha
                        or report.get('bundle_sha') != self.bundle_sha
                        or report.get('config_sha') != self.config_sha):
                    raise ShelfRejected('Task Planner protocol/code/config handshake failed')
                if report.get('time_monotonic', 0) >= sent:
                    if report.get('accepted') is False:
                        self.log(dict(event='shelf_planner_rejected', **report))
                        raise ShelfRejected('Task Planner rejected: '+str(report.get('error')))
                    if report.get('stage') == stage and report.get('accepted') is True:
                        self.log(dict(event='shelf_planner', **report))
                        return report
            getattr(self.robot, 'operator_sleep', time.sleep)(.02)
        raise ShelfRejected('Task Planner timed out; no executable fallback')

    def _request(self, operation, target=None, timeout=20., **data):
        self.checkpoint()
        if target is None: target = self._pose()[2]
        target = transform(target, 'task Planner target')
        ticket = uuid.uuid4().hex
        request = dict(protocol=PROTOCOL, ticket=ticket, operation=operation,
            bundle_sha=self.bundle_sha, config_sha=self.config_sha, planner_sha=self.planner_sha,
            created_monotonic=time.monotonic(), target_tcp=target.tolist(), **data)
        path = ipc_path(ticket, 'request'); path.parent.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream: json.dump(request, stream, allow_nan=False)
        sent = time.monotonic()
        self.robot.udp.send(self._command('__preview__:book_task:'+ticket, target))
        try:
            report = self._wait(ticket, sent, operation, timeout)
            if operation == 'plan': self.pending[ticket] = target.copy()
            return report
        finally:
            path.unlink(missing_ok=True)
            if operation != 'plan': ipc_path(ticket, 'report').unlink(missing_ok=True)

    def probe(self):
        """Read-only protocol/code handshake; performs no GPU initialization."""
        return self._request('probe')

    def update_scene(self, scene, *, omit_names=()):
        result = self._request('scene', timeout=180., scene=scene.to_dict(), omit_names=list(omit_names))
        self.scene_version = scene.version; self.pending.clear()
        return result

    def attach(self, geometry, book_id):
        result = self._request('attach', scene_version=self.scene_version,
                               geometry=asdict(geometry), book_id=book_id)
        self.pending.clear(); return result

    def detach(self, book_id, *, released):
        result = self._request('detach', scene_version=self.scene_version,
                               book_id=book_id, released=bool(released))
        self.pending.clear(); return result

    def plan_segment(self, tcp_pose, stage, scene_version=None, *, linear=False):
        return self._request('plan', target=tcp_pose, timeout=35.,
            scene_version=scene_version or self.scene_version, segment_stage=stage, linear=bool(linear))

    def validate_sequence(self, waypoints, geometry, book_id, *, contact_book):
        """Virtual carrying preflight only; does not authorize any actual motion."""
        return self._request('validate_sequence', timeout=150., scene_version=self.scene_version,
            waypoints=list(waypoints), geometry=asdict(geometry), book_id=book_id,
            contact_book=dict(contact_book))

    def execute(self, ticket):
        if isinstance(ticket, dict): ticket = ticket['ticket']
        target = self.pending.pop(ticket, None)
        if target is None: raise ShelfRejected('Unknown/consumed task trajectory')
        self.checkpoint()
        command = self._command('book_task_execute:'+ticket, target)
        self.robot.target_command = copy.deepcopy(command)
        sent = time.monotonic(); self.robot.udp.send(command)
        report = self._wait(ticket, sent, 'execute', 20.)
        from baseline_approach_client import joint_arrival
        joint_arrival(self.robot, np.asarray(report['target_joint_rad']), self.checkpoint,
                      timeout=min(90., float(report['duration_s'])+15.))
        return report

    def hold(self, ticket):
        if isinstance(ticket, dict): ticket = ticket['ticket']
        self.checkpoint()
        target = self._pose()[2]
        command = self._command('book_task_hold:'+ticket, target)
        self.robot.target_command = copy.deepcopy(command)
        sent = time.monotonic(); self.robot.udp.send(command)
        try:
            return self._wait(ticket, sent, 'hold', 10.)
        finally:
            ipc_path(ticket, 'report').unlink(missing_ok=True)

    def close(self):
        result = self._request('close', scene_version=self.scene_version)
        self.pending.clear(); self.scene_version = None
        return result
