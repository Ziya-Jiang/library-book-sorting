"""Scene-version-bound shelf planning on an isolated, lazy CuRobo engine.

The original single-book MotionGen/world are never swapped or edited. Preview
requests contain geometry only. Joint paths remain in this process and expire.
"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

AGENT_ROOT = Path(__file__).resolve().parents[3]/'dual_v2_2_agent'
if str(AGENT_ROOT) not in sys.path: sys.path.insert(0, str(AGENT_ROOT))
from book_tasks.planner_client import ipc_path, planner_code_digest, capability_path, process_start, PROTOCOL
from book_tasks.planner_scene import CuroboSceneSession, ATTACHMENT_LINK, GRASP_CONTACT_LINKS
from book_tasks.placement import HeldGeometry
from book_tasks.shelf import ShelfRejected, ShelfScene, Box, transform
from baseline_approach_planner import Adapter

LOADED_PLANNER_SHA = planner_code_digest()


def write_capability(planner):
    """Advertise loaded code only; never initialize the optional task GPU engine."""
    if not hasattr(planner, '_book_tasks_motion_options'):
        raise ShelfRejected('Task Planner construction options unavailable')
    if getattr(planner, '_motion_gen', None) is None:
        return  # An IK-only process cannot advertise a trajectory capability.
    path = capability_path(); path.parent.mkdir(mode=0o700, exist_ok=True)
    data = dict(protocol=PROTOCOL, planner_sha=LOADED_PLANNER_SHA, pid=os.getpid(),
        start_ticks=process_start(os.getpid()), port=int(os.environ.get('CUARM_V31_TARGET_LOCAL_PORT', '10176')),
        capabilities=['shelf_scene_v1', 'attached_right_book', 'rt_curve_check'], task_engine_initialized=False)
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(data)+'\n')
    os.chmod(temporary, 0o600); temporary.replace(path)


def decode(description):
    match = re.fullmatch(r'(__preview__:book_task|book_task_execute|book_task_hold):([a-f0-9]{32})', description)
    return (match[1], match[2]) if match else None


def isolated_motion_gen(planner):
    """Copy the installed robot and solver settings, adding only task capacity."""
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    cfg = copy.deepcopy(planner.robot_cfg)
    kin = cfg['kinematics']
    right = planner.target_link_names[1]
    if right != 'arm_end_effector_r':
        raise ShelfRejected('Unsupported right attachment parent; inspect the actual robot contract')
    kin['extra_links'] = dict(kin.get('extra_links') or {})
    if ATTACHMENT_LINK in kin['extra_links']:
        raise ShelfRejected('Dedicated task attachment link already exists in base profile')
    kin['extra_links'][ATTACHMENT_LINK] = dict(link_name=ATTACHMENT_LINK,
        joint_name='book_tasks_attachment_fixed', joint_type='FIXED',
        fixed_transform=[0, 0, 0, 1, 0, 0, 0], parent_link_name=right)
    kin['extra_collision_spheres'] = dict(kin.get('extra_collision_spheres') or {})
    kin['extra_collision_spheres'][ATTACHMENT_LINK] = 128
    kin['collision_link_names'] = list(kin['collision_link_names'])+[ATTACHMENT_LINK]
    # No guessed collision-ignore links. A held book may contact configured
    # right gripper pads only; the exact list must come from the robot profile.
    contact = list(GRASP_CONTACT_LINKS)
    if any(link not in kin['collision_link_names'] or link.endswith('_l') for link in contact):
        raise ShelfRejected('Invalid measured right-gripper contact-link profile')
    ignore = copy.deepcopy(kin.get('self_collision_ignore') or {})
    ignore[ATTACHMENT_LINK] = contact
    kin['self_collision_ignore'] = ignore
    options = copy.deepcopy(planner._book_tasks_motion_options)
    options['collision_cache'] = dict(obb=256, mesh=max(16, len(getattr(planner.world_cfg, 'mesh', []) or [])))
    with planner._gpu_lock:
        robot = RobotConfig.from_dict(cfg, planner.tensor_args)
        motion = MotionGen(MotionGenConfig.load_from_robot_config(robot,
            copy.deepcopy(planner.world_cfg), planner.tensor_args, **options))
    return motion


def pose_error(actual, target):
    return (float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            float((Rotation.from_matrix(actual[:3, :3])*Rotation.from_matrix(target[:3, :3]).inv()).magnitude()))


def validate_linear(positions, quaternions, start, end):
    """Check the complete 1ms RT curve, including arbitrary shelf directions."""
    points = np.asarray(positions); delta = end[:3, 3]-start[:3, 3]
    length = np.linalg.norm(delta)
    if length < 1e-6: fraction = np.zeros(len(points))
    else: fraction = (points-start[:3, 3])@delta/(length*length)
    projected = start[:3, 3]+np.clip(fraction, 0, 1)[:, None]*delta
    if np.max(np.linalg.norm(points-projected, axis=1)) > .003:
        raise ShelfRejected('RT TCP path left the 3mm straight-segment corridor')
    if length > .003 and np.any(np.diff(fraction)*length < -.003):
        raise ShelfRejected('RT straight segment reverses insertion direction')
    rotations = Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]]))
    expected = Slerp([0., 1.], rotations)(np.clip(fraction, 0, 1))
    actual = Rotation.from_quat(np.asarray(quaternions)[:, [1, 2, 3, 0]])
    if np.max((actual*expected.inv()).magnitude()) > np.deg2rad(1.):
        raise ShelfRejected('RT straight segment exceeded 1 degree pose corridor')


class Transactions:
    def __init__(self, planner):
        self.p = planner; self.scene = None; self.records = {}; self.used = set()
        self.request_identities = {}
        self.last_description = None; self.last_session = None; self.failed = False
        self.bundle_sha = None; self.config_sha = None
        # Hash the loaded implementation now. A later filesystem deployment
        # cannot make this old process masquerade as the new Planner version.
        self.planner_sha = LOADED_PLANNER_SHA
        self.bound_identity = None

    def write(self, ticket, stage, accepted, **data):
        path = ipc_path(ticket, 'report'); path.parent.mkdir(mode=0o700, exist_ok=True)
        bundle_sha, config_sha = self.request_identities.get(ticket, (self.bundle_sha, self.config_sha))
        data.update(protocol=PROTOCOL, ticket=ticket, stage=stage, accepted=accepted,
            planner_pid=os.getpid(), time_monotonic=time.monotonic(), planner_sha=self.planner_sha,
            bundle_sha=bundle_sha, config_sha=config_sha)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, allow_nan=False,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x))+'\n')
        temporary.replace(path)

    def adapter(self, snapshot):
        a = Adapter(self.p); a.session = getattr(snapshot['panel'], '_seldon_session', None)
        a.description = self.p.target_description
        if a.session is None or np.max(np.abs(snapshot['velocity'])) > .02:
            raise ShelfRejected('Shelf planning requires stationary, measured session feedback')
        a.check(); return a

    def read(self, ticket):
        if ticket in self.used: raise ShelfRejected('Consumed task request identity')
        path = ipc_path(ticket, 'request')
        fd = os.open(path, os.O_RDONLY|getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_size > 2*1024*1024
                    or info.st_uid != os.getuid() or info.st_mode & 0o022):
                raise ShelfRejected('Unsafe or oversized task request file')
            data = json.load(stream)
        if data.get('ticket') != ticket or data.get('protocol') != PROTOCOL:
            raise ShelfRejected('Invalid task request protocol/identity')
        identity = (data.get('bundle_sha'), data.get('config_sha'))
        if any(not re.fullmatch('[a-f0-9]{64}', str(x)) for x in identity):
            raise ShelfRejected('Missing task bundle/config handshake')
        if data.get('planner_sha') != self.planner_sha:
            raise ShelfRejected('Planner code changed; restart in the maintenance window')
        age = time.monotonic()-data.get('created_monotonic', -1e9)
        if not 0 <= age <= 10.: raise ShelfRejected('Expired task geometry request')
        self.used.add(ticket)
        self.request_identities[ticket] = identity
        if len(self.used) > 4096: raise ShelfRejected('Task request capacity exhausted; restart task session')
        return data

    def ensure_scene(self, request, adapter):
        if self.scene is None or self.scene.session != adapter.session:
            raise ShelfRejected('No matching task scene/control session')
        if request.get('scene_version') != self.scene.scene.version:
            raise ShelfRejected('Stale task scene version; observe and plan again')
        return self.scene.ticket()

    def link_to_tcp(self):
        t = np.eye(4)
        offsets = np.asarray(self.p.end_effector_offset, float)
        if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
            raise ShelfRejected('Invalid installed TCP calibration')
        t[:3, 3] = offsets[1]
        return t

    def tcp(self, joints):
        pos, quat = self.p._v31_bridge.tcp(joints)
        t = np.eye(4); t[:3, 3] = pos[0, 1]
        t[:3, :3] = Rotation.from_quat(quat[0, 1, [1, 2, 3, 0]]).as_matrix()
        return t

    def plan(self, adapter, start, target, *, linear, deadline):
        from curobo.types.state import JointState
        from curobo.types.math import Pose
        from cuarm_utils import remove_tool_offset
        p = self.p; adapter.check(); ticket = self.scene.ticket()
        positions, orientations = p._v31_bridge.tcp(start)
        positions, orientations = positions[0].copy(), orientations[0].copy()
        positions[1] = target[:3, 3]
        orientations[1] = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        values = p.tensor_args.to_device(adapter.arm_values(start))
        state = JointState(position=values, velocity=values*0, acceleration=values*0, jerk=values*0,
                           joint_names=p.curobo_input_joint_names)
        links = {name: Pose(position=p.tensor_args.to_device(remove_tool_offset(positions[i],
            orientations[i], p.end_effector_offset[i])), quaternion=p.tensor_args.to_device(orientations[i]))
            for i, name in enumerate(p.target_link_names)}
        config = p._plan_config.clone(); config.max_attempts = 1
        config.timeout = max(.01, deadline-time.monotonic()); config.check_start_validity = True
        config.time_dilation_factor = getattr(p, '_baseline_speed_factor', None) or .4
        with p._gpu_lock:
            result = self.scene.motion.plan_single(state, links[p.target_link_names[p.ee_link_idx_in_target_link_names]],
                                                   config, link_poses=links)
        if time.monotonic() >= deadline: raise ShelfRejected('Shelf planning deadline expired')
        adapter.check(); self.scene.require(ticket)
        if not bool(result.success.item()): raise ShelfRejected('Shelf MotionGen: '+str(result.status))
        points = result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy()
        # Preserve the measured float64 hold before the exact left-arm check.
        points = points.astype(np.float64, copy=True).reshape(-1, 14)
        points[:, :7] = start[:7]
        dt = float(result.optimized_dt.detach().cpu().numpy().item())
        if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(dt) or dt <= 0:
            raise ShelfRejected('Invalid task MotionGen path')
        for _ in range(6):
            samples = adapter.sample(start, points, dt, .001)
            ratios = adapter.dynamics(samples, .001)
            if np.all(ratios <= 1.): break
            dt *= max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))*1.1
        else: raise ShelfRejected('Task RT path dynamics exceed unchanged robot limits')
        def checkpoint():
            adapter.check()
            if time.monotonic() >= deadline: raise ShelfRejected('Task RT validation deadline expired')
        self.validate(adapter, samples, start, target, linear, ticket, checkpoint)
        return dict(points=points.copy(), dt=dt, start=start.copy(), target=target.copy(),
                    linear=linear, scene_ticket=ticket, grip=adapter.grip,
                    duration_s=float(len(points)*dt))

    def validate(self, adapter, samples, start, target, linear, ticket, checkpoint):
        if (not np.allclose(samples[:, :7], start[:7], atol=1e-8, rtol=0)
                or np.any(samples < adapter.bounds[0]) or np.any(samples > adapter.bounds[1])):
            raise ShelfRejected('Task path changes left arm or exceeds unchanged joint limits')
        if np.any(adapter.dynamics(samples, .001) > 1.):
            raise ShelfRejected('Task splice dynamics failed')
        self.scene.check_samples(samples, adapter, checkpoint, ticket)
        error = pose_error(self.tcp(samples[-1]), target)
        if error[0] > .003 or error[1] > np.deg2rad(1.):
            raise ShelfRejected('Task path failed measured TCP endpoint check')
        if linear:
            positions, orientations = self.p._v31_bridge.tcp(samples)
            validate_linear(positions[:, 1], orientations[:, 1], self.tcp(start), target)

    def sequence(self, request, adapter, snapshot):
        ticket = self.ensure_scene(request, adapter)
        if self.scene.attached_book: raise ShelfRejected('Pre-grasp sequence requires an empty gripper model')
        waypoints = request.get('waypoints', [])
        if not 1 <= len(waypoints) <= 16: raise ShelfRejected('Invalid shelf sequence length')
        geometry = HeldGeometry(**request['geometry']); book_id = str(request['book_id'])
        original = self.scene.scene; original_omit = tuple(getattr(self, 'omit_names', ()))
        contact = Box(**request['contact_book'])
        if contact.kind != 'book' or contact.name != book_id or contact not in original.obstacles:
            raise ShelfRejected('Contact book is not the measured selected obstacle in this scene')
        omitted = (contact.name,)
        self.check_omissions(original, omitted)
        deadline = time.monotonic()+120.; summaries = []
        try:
            self.scene.update(original, adapter.session, omit_names=omitted)
            start = snapshot['joints'].copy()
            previous_center = None
            attached_once = False; released_once = False
            for waypoint in waypoints:
                if waypoint['attachment'] == 'held' and self.scene.attached_book is None:
                    if attached_once or released_once:
                        raise ShelfRejected('Sequence cannot grasp twice or reattach a released book')
                    actual_book = self.tcp(start) @ np.asarray(geometry.tcp_to_book)
                    if np.linalg.norm(original.frame.from_base(actual_book[:3, 3])-contact.center) > .003:
                        raise ShelfRejected('Virtual grasp does not match the measured selected book')
                    self.scene.attach(geometry, book_id, self.link_to_tcp(), self.scene.ticket())
                    attached_once = True
                elif waypoint['attachment'] == 'released' and self.scene.attached_book:
                    if previous_center is None: raise ShelfRejected('Release before a measured placement')
                    self.scene.detach(book_id, self.scene.ticket(), released=True)
                    dims = np.asarray(geometry.book_dimensions_m); center = np.asarray(previous_center)
                    placed = Box('preflight_placed_'+book_id, (center-dims*.5).tolist(), (center+dims*.5).tolist(), 'book')
                    updated = replace(original, obstacles=original.obstacles+(placed,), version='')
                    self.scene.update(updated, adapter.session, omit_names=omitted)
                    released_once = True
                elif waypoint['attachment'] != 'held' and waypoint['attachment'] != 'released':
                    raise ShelfRejected('Invalid sequence attachment transition')
                linear = waypoint.get('linear', waypoint['stage'] != 'target_prepose')
                path = self.plan(adapter, start, transform(waypoint['tcp_pose_base']),
                                 linear=bool(linear), deadline=deadline)
                summaries.append(dict(stage=waypoint['stage'], duration_s=path['duration_s'], checked=True))
                start = path['points'][-1].copy()
                actual_book = self.tcp(start) @ np.asarray(geometry.tcp_to_book)
                previous_center = original.frame.from_base(actual_book[:3, 3]).tolist()
            if not attached_once or not released_once:
                raise ShelfRejected('Sequence must include grasp, attached transfer, release and retreat')
            return dict(segments=summaries, virtual_only=True, executable=False, scene_version=original.version)
        finally:
            if self.scene.attached_book:
                self.scene.detach(book_id, self.scene.ticket(), released=True)
            self.scene.update(original, adapter.session, omit_names=original_omit)
            self.records.clear()

    @staticmethod
    def check_omissions(scene, names):
        books = {box.name for box in scene.obstacles if box.kind == 'book'}
        if len(names) > 1 or any(name not in books for name in names):
            raise ShelfRejected('Only the explicitly selected book may be removed from obstacle world')

    def request(self, ticket, snapshot):
        data = self.read(ticket); operation = data['operation']
        if operation == 'probe':
            if np.max(np.abs(snapshot['velocity'])) > .02:
                raise ShelfRejected('Capability probe requires a stationary measured hold')
            self.write(ticket, operation, True, capabilities=['scene', 'attached_right_book', 'rt_curve_check',
                'linear_segments', 'sequence_preflight'], execution_enabled=False); return
        adapter = self.adapter(snapshot)
        identity = (adapter.session, data['bundle_sha'], data['config_sha'])
        if (self.bound_identity is not None and self.bound_identity != identity
                and self.bound_identity[0] != adapter.session and operation == 'scene'
                and self.scene is not None and self.scene.attached_book is None):
            # A cancelled empty-handed batch must not permanently own the
            # process. The new control session already proved stationary fresh
            # feedback; only a new scene may abandon its obsolete world.
            self.scene.close(robot_stopped=True)
            self.records.clear(); self.bound_identity = None
        if self.bound_identity is not None and self.bound_identity != identity:
            raise ShelfRejected('Task session/bundle/config changed; close the prior scene first')
        self.bound_identity = identity
        self.bundle_sha, self.config_sha = identity[1:]
        target = transform(data['target_tcp'])
        wire = np.eye(4); wire[:3, 3] = self.p.target_arm_tool_position[1]
        wire[:3, :3] = Rotation.from_quat(np.asarray(self.p.target_arm_tool_orientation[1])[[1, 2, 3, 0]]).as_matrix()
        if max(pose_error(target, wire)) > 1e-6: raise ShelfRejected('Task wire/geometry target mismatch')
        held_error = pose_error(target, self.tcp(snapshot['joints']))
        if operation != 'plan' and (held_error[0] > .003 or held_error[1] > np.deg2rad(1.)):
            raise ShelfRejected('Nonmotion task request is not a measured hold')
        guard = getattr(self.p._v31_bridge, 'guard', None)
        if guard is not None and guard.fault:
            raise ShelfRejected('Task preview refused a faulted robot guard')
        self.records.clear()
        self.p._panel_begin_planning(snapshot['joints'].reshape(2, 7).tolist())
        if operation == 'scene':
            scene = ShelfScene.from_dict(data['scene']); self.check_omissions(scene, data.get('omit_names', ()))
            if self.scene is None or self.scene.closed:
                motion = isolated_motion_gen(self.p)
                view = SimpleNamespace(_motion_gen=motion, _gpu_lock=self.p._gpu_lock,
                    world_cfg=copy.deepcopy(self.p.world_cfg), tensor_args=self.p.tensor_args)
                self.scene = CuroboSceneSession(view)
            self.scene.update(scene, adapter.session, omit_names=data.get('omit_names', ()))
            self.omit_names = tuple(data.get('omit_names', ()))
            output = dict(scene_version=scene.version)
        else:
            stamp = self.ensure_scene(data, adapter)
            if operation == 'attach':
                contacts = self.scene.contact_books
                geometry = HeldGeometry(**data['geometry'])
                measured = self.tcp(snapshot['joints']) @ np.asarray(geometry.tcp_to_book)
                if (len(contacts) != 1 or contacts[0].name != data['book_id']
                        or np.linalg.norm(self.scene.scene.frame.from_base(measured[:3, 3])-contacts[0].center) > .005
                        or np.max(np.abs(np.asarray(geometry.book_dimensions_m)-contacts[0].dimensions)) > .003):
                    raise ShelfRejected('Attachment does not match the selected measured grasp target')
                output = self.scene.attach(geometry, data['book_id'], self.link_to_tcp(), stamp)
            elif operation == 'detach':
                output = self.scene.detach(data['book_id'], stamp, released=data.get('released') is True)
            elif operation == 'close':
                self.scene.close(robot_stopped=True); self.bound_identity = None; output = dict(closed=True)
            elif operation == 'validate_sequence':
                output = self.sequence(data, adapter, snapshot)
            elif operation == 'plan':
                path = self.plan(adapter, snapshot['joints'], target, linear=data.get('linear') is True,
                                 deadline=time.monotonic()+25.)
                adapter.check()
                newest = self.p._v31_bridge.snapshot(max_age=.25)
                if np.max(np.abs(newest['joints']-snapshot['joints'])) > .002:
                    raise ShelfRejected('Start moved during task preview')
                path.update(expires=time.monotonic()+30., phase='ready', session=adapter.session,
                            bundle_sha=self.bundle_sha, config_sha=self.config_sha)
                self.records[ticket] = path
                output = dict(scene_version=self.scene.scene.version, duration_s=path['duration_s'],
                              segment_stage=data['segment_stage'], linear=path['linear'])
            else: raise ShelfRejected('Unknown task Planner operation')
        adapter.check(); self.write(ticket, operation, True, **output)

    def dispatch(self, ticket, phase, snapshot):
        p = self.p; record = self.records.get(ticket)
        if not record or time.monotonic() > record['expires']:
            raise ShelfRejected('Unknown/expired task trajectory')
        adapter = self.adapter(snapshot)
        if adapter.session != record['session']: raise ShelfRejected('Task control session changed')
        if (record['bundle_sha'], record['config_sha']) != (self.bundle_sha, self.config_sha):
            raise ShelfRejected('Task trajectory bundle/config identity changed')
        self.scene.require(record['scene_ticket'])
        if phase == 'book_task_hold':
            error = pose_error(self.tcp(snapshot['joints']), record['target'])
            if record['phase'] != 'executing' or error[0] > .003 or error[1] > np.deg2rad(1.):
                raise ShelfRejected('Task hold before confirmed endpoint')
            held = snapshot['joints'].reshape(2, 7).tolist(); p._cmd_plan = None
            p.curobo_cmd.arm_joint_command = held; p.curobo_cmd.waypoints = [held, held]
            p.curobo_cmd.waypoints_size = 2; p.curobo_cmd.waypoints_dt = .2; p.curobo_cmd.cmd_plan_size = 0
            record['phase'] = 'holding'; record['hold_target'] = np.c_[p.target_arm_tool_position, p.target_arm_tool_orientation].copy()
            self.write(ticket, 'hold', True, no_trajectory_replay=True); return
        target = record['target']; wireq = Rotation.from_quat(np.asarray(p.target_arm_tool_orientation[1])[[1, 2, 3, 0]])
        if (record['phase'] != 'ready' or np.linalg.norm(p.target_arm_tool_position[1]-target[:3, 3]) > 1e-8
                or (wireq*Rotation.from_matrix(target[:3, :3]).inv()).magnitude() > 1e-6
                or list(p.target_gripper_joint) != record['grip']
                or np.max(np.abs(snapshot['joints']-record['start'])) > .002):
            raise ShelfRejected('Task ticket consumed or target/gripper/start identity changed')
        p._panel_begin_planning(snapshot['joints'].reshape(2, 7).tolist())
        points = record['points'].copy()
        # The measured left hold can differ by encoder quantization from the
        # preview. Keep it exactly stationary and revalidate this entire splice.
        points[:, :7] = snapshot['joints'][:7]
        samples = adapter.sample(snapshot['joints'], points, record['dt'], .001)
        self.validate(adapter, samples, snapshot['joints'], target, record['linear'], record['scene_ticket'], adapter.check)
        newest = p._v31_bridge.snapshot(max_age=.25)
        if np.max(np.abs(newest['joints']-snapshot['joints'])) > .002:
            raise ShelfRejected('Start moved during final task validation')
        from curobo.types.state import JointState
        record['points'] = points
        p._move_arm = [False, True]; p.waypoints = points.copy(); p.limited_waypoints = points.copy()
        p._prev_arm_cmd_pos = snapshot['joints'].copy()
        p.waypoints_dt = record['dt']; p._cmd_idx = 0
        p._cmd_plan = JointState(position=p.tensor_args.to_device(samples), joint_names=p.total_arm_joint_names)
        p.curobo_cmd.waypoints = record['points'].reshape(-1, 2, 7).tolist()
        p.curobo_cmd.waypoints_size = len(record['points']); p.curobo_cmd.waypoints_dt = record['dt']
        p.curobo_cmd.cmd_plan_size = len(samples); p._panel_replan_pending = False
        record['phase'] = 'executing'; record['expires'] = time.monotonic()+record['duration_s']+30.
        self.write(ticket, 'execute', True, target_joint_rad=record['points'][-1], duration_s=record['duration_s'])

    def compute(self):
        from cuarm_state import CuroboStatus
        p = self.p; phase, ticket = decode(p.target_description)
        try:
            snapshot = p._v31_bridge.snapshot(max_age=.25)
            p.arm_joint_position = snapshot['joints'].copy()
            p.arm_joint_velocity = snapshot['velocity'].copy()
            session = getattr(snapshot['panel'], '_seldon_session', None)
            guard = getattr(p._v31_bridge, 'guard', None)
            if guard is not None and guard.fault and phase != '__preview__:book_task':
                raise ShelfRejected('Task refused a faulted robot guard')
            if self.last_description == p.target_description:
                if session != self.last_session or getattr(p, '_panel_replan_pending', False):
                    raise ShelfRejected('Task session/start changed during execution')
                record = self.records.get(ticket)
                if record:
                    self.scene.require(record['scene_ticket'])
                    if record['phase'] == 'holding':
                        if not np.allclose(np.c_[p.target_arm_tool_position, p.target_arm_tool_orientation], record['hold_target'], atol=1e-8, rtol=0):
                            raise ShelfRejected('Task hold identity changed')
                        p.set_gripper_command(); return True
                    if list(p.target_gripper_joint) != record['grip']:
                        raise ShelfRejected('Gripper changed during task trajectory')
                    if record['phase'] == 'executing':
                        current_q = Rotation.from_quat(np.asarray(p.target_arm_tool_orientation[1])[[1, 2, 3, 0]])
                        if (np.linalg.norm(p.target_arm_tool_position[1]-record['target'][:3, 3]) > 1e-8
                                or (current_q*Rotation.from_matrix(record['target'][:3, :3]).inv()).magnitude() > 1e-6):
                            raise ShelfRejected('Task target changed during execution')
                        p._get_next_planned_arm_cmd()
                return True
            self.last_description = p.target_description; self.last_session = session; self.failed = False
            p._clear_planned_result(); p._hold_current_arm_command()
            p.status = CuroboStatus.PLANNING; p._curobo_planning_status_planning_count = 0
            if phase == '__preview__:book_task': self.request(ticket, snapshot)
            else:
                self.dispatch(ticket, phase, snapshot)
                if phase == 'book_task_execute': p._get_next_planned_arm_cmd()
            p.status = CuroboStatus.PLANNING_SUCCEEDED
            p._prev_target_tool_position = p.target_arm_tool_position.copy()
            p._prev_target_tool_orientation = p.target_arm_tool_orientation.copy()
            p._prev_target_arm_joint = snapshot['joints'].copy(); p.set_gripper_command()
        except Exception as error:
            self.failed = True; self.records.clear()
            p._clear_planned_result(); p._hold_current_arm_command(); p.status = CuroboStatus.PLANNING_FAILED
            self.write(ticket, phase, False, error=f'{type(error).__name__}: {error}')
        return True


def compute(planner):
    if decode(planner.target_description) is None:
        current = getattr(planner, '_book_task_transactions', None)
        if current is not None:
            was_executing = any(record['phase'] == 'executing' for record in current.records.values())
            current.records.clear(); current.last_description = None
            if was_executing or planner.target_description == 'book_operator_hold':
                planner._clear_planned_result(); planner._hold_current_arm_command()
        return False
    if not hasattr(planner, '_book_task_transactions'):
        planner._book_task_transactions = Transactions(planner)
    return planner._book_task_transactions.compute()
