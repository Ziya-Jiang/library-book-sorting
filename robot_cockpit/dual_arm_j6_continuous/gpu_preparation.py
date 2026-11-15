"""Read-only J6 preparation IK in the running CuRobo Planner's CUDA context.

The socket never submits a motion command. The caller still validates every
candidate and sends accepted motion through the ordinary guarded control path.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parent
DIRECTORY = Path('/tmp') / f'j6-continuous-preparation-ik-{os.getuid()}'
MAX_BYTES = 262144


def socket_path(pid):
    return DIRECTORY / f'{int(pid)}.sock'


def read_message(sock):
    data = bytearray()
    while b'\n' not in data:
        part = sock.recv(4096)
        if not part:
            raise RuntimeError('J6 GPU IK connection closed')
        data.extend(part)
        if len(data) > MAX_BYTES:
            raise ValueError('J6 GPU IK response too large')
    line, tail = bytes(data).split(b'\n', 1)
    if tail:
        raise ValueError('Unexpected extra GPU IK message')
    return json.loads(line)


def send_message(sock, value):
    data = json.dumps(value, allow_nan=False).encode()+b'\n'
    if len(data) > MAX_BYTES:
        raise ValueError('J6 GPU IK message too large')
    sock.sendall(data)


def solve_candidates(model, joints, grippers, position, quaternion, seed, seconds, *, connection_goal=None):
    from start import existing_planner
    from j6_only import require_capability, revision
    from planning_receipt import process_stamp
    pid = existing_planner()
    require_capability(pid)
    request = dict(request_id=uuid.uuid4().hex, planner_pid=pid,
                   planner_start=process_stamp(pid), revision=revision(), model_digest=model.digest,
                   joints=np.asarray(joints).tolist(), grippers=np.asarray(grippers).tolist(),
                   position=np.asarray(position).tolist(), quaternion=np.asarray(quaternion).tolist(),
                   seed=np.asarray(seed).tolist(), deadline=time.monotonic()+max(0., seconds))
    if connection_goal is not None:
        request.update(operation='preparation_connection_v1', goal_joints=np.asarray(connection_goal).tolist())
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(max(.05, seconds)+.2)
            sock.connect(str(socket_path(pid)))
            send_message(sock, request)
            reply = read_message(sock)
    except (OSError, ValueError) as error:
        raise RuntimeError('J6 CuRobo GPU准备求解不可用；不回退CPU优化：'+str(error)) from error
    if (reply.get('request_id') != request['request_id'] or reply.get('planner_pid') != pid
            or reply.get('planner_start') != request['planner_start']
            or reply.get('revision') != request['revision']
            or process_stamp(pid) != request['planner_start']):
        raise RuntimeError('J6 GPU准备响应的请求或Planner世代不匹配')
    if reply.get('error'):
        raise RuntimeError('J6 CuRobo GPU准备求解失败：'+reply['error'])
    if reply.get('expired'):
        if connection_goal is not None:return dict(accepted=False,reason='connection_deadline')
        return [], dict(solver='CuRobo GPU IK', budget_exhausted=True)
    if connection_goal is not None:
        return reply['diagnostics']
    return reply['candidates'], reply['diagnostics']


@contextmanager
def preparation_bounds(solver, arm_names, bounds, joints, grip_values):
    """Scoped, in-place bounds under the shared GPU lock; always restore them.

    Hold left arm and grippers; right J1-J7 are free during preparation.
    J7's actual later rotation remains a separate single-joint action.
    """
    saved = {}
    fixed = dict(zip(arm_names[:7], joints[:7]))
    fixed.update(grip_values)
    try:
        for rollout in solver.get_all_rollout_instances():
            names = list(rollout.joint_names)
            for attr in ('bound_cost', 'bound_constraint'):
                tensor = getattr(getattr(getattr(rollout, attr, None), 'joint_limits', None), 'position', None)
                if tensor is None:
                    continue
                key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
                if key in saved:
                    continue
                original = tensor.clone()
                desired = original.clone()
                for index, name in enumerate(names):
                    if name in fixed:
                        lo, hi = fixed[name]-1e-6, fixed[name]+1e-6
                    elif name in arm_names[7:]:
                        j = arm_names.index(name)
                        lo, hi = bounds[:, j] + np.array([1., -1.])*np.deg2rad(2.)
                    else:
                        raise ValueError('Unknown free joint in J6 preparation IK: '+name)
                    desired[0, index] = max(float(original[0, index]), float(lo))
                    desired[1, index] = min(float(original[1, index]), float(hi))
                    if desired[0, index] >= desired[1, index]:
                        raise ValueError('Empty GPU preparation joint bounds: '+name)
                saved[key] = tensor, original
                tensor.copy_(desired)
        if not saved:
            raise RuntimeError('CuRobo preparation bounds unavailable')
        from lift_solver_bounds import synchronized_ik_bounds
        with synchronized_ik_bounds(solver):
            yield
    finally:
        for tensor, original in saved.values():
            tensor.copy_(original)


class PreparationService:
    def __init__(self, planner):
        from backend.board_servo.model import Model
        from j6_only import revision
        from planning_receipt import process_stamp
        self.planner = planner
        self.model = Model(ROOT/'backend/board_servo')
        from axis_objective import build_axis_solver, warm_axis_solver
        self.axis_solver = build_axis_solver(planner, self.model)
        warm_axis_solver(self.axis_solver)
        self.identity = dict(planner_pid=os.getpid(), planner_start=process_stamp(os.getpid()), revision=revision())
        self.stopped = threading.Event()
        DIRECTORY.mkdir(mode=0o700, exist_ok=True)
        if DIRECTORY.is_symlink() or DIRECTORY.stat().st_uid != os.getuid():
            raise RuntimeError('Invalid GPU preparation socket directory')
        DIRECTORY.chmod(0o700)
        self.path = socket_path(os.getpid())
        self.path.unlink(missing_ok=True)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.path)); self.socket.listen(2); self.socket.settimeout(.25)
        self.thread = threading.Thread(target=self.serve, name='j6-readonly-gpu-ik', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.socket.close()
        self.path.unlink(missing_ok=True)

    def serve(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with client:
                client.settimeout(1.)
                request = {}
                try:
                    value = read_message(client)
                    if not isinstance(value, dict):
                        raise ValueError('GPU IK request must be an object')
                    request = value
                    reply = self.solve(request)
                except Exception as error:
                    reply = dict(error=f'{type(error).__name__}: {error}')
                reply.update(self.identity, request_id=request.get('request_id'))
                try:
                    send_message(client, reply)
                except OSError:
                    pass

    def solve(self, request):
        for key, value in self.identity.items():
            if request.get(key) != value:
                raise ValueError('GPU IK process/code contract mismatch: '+key)
        if request.get('model_digest') != self.model.digest:
            raise ValueError('GPU IK CPU validation model mismatch')
        if request.get('operation') == 'preparation_connection_v1':
            from preparation_connection import serve
            return serve(self, request)
        if request.get('operation') is not None:
            from lift_preflight import OPERATION, solve
            if request['operation'] != OPERATION:
                raise ValueError('Unknown read-only preparation operation')
            return solve(self, request)
        q = np.asarray(request['joints'], float)
        grip = np.asarray(request['grippers'], float)
        position = np.asarray(request['position'], float)
        quat = np.asarray(request['quaternion'], float)
        seed = np.asarray(request['seed'], float)
        deadline = float(request['deadline'])
        if (q.shape != (14,) or seed.shape != (14,) or grip.shape != (2,)
                or position.shape != (3,) or quat.shape != (4,)
                or not all(np.isfinite(x).all() for x in (q, seed, grip, position, quat))
                or abs(np.linalg.norm(quat)-1.) > 1e-6
                or np.any(q < self.model.bounds[0]) or np.any(q > self.model.bounds[1])
                or np.any(grip < 0) or np.any(grip > 1)
                or not np.isfinite(deadline) or deadline-time.monotonic() > 9.):
            raise ValueError('Invalid GPU preparation request')
        remaining = deadline-time.monotonic()
        if remaining <= 0 or not self.planner._gpu_lock.acquire(timeout=max(0., remaining)):
            return dict(expired=True)
        try:
            if self.stopped.is_set() or time.monotonic() >= deadline:
                return dict(expired=True)
            return self.solve_locked(q, grip, position, quat, seed, deadline)
        finally:
            self.planner._gpu_lock.release()

    def solve_locked(self, q, grip, position, quat, seed, deadline):
        import torch
        from curobo.types.math import Pose
        from j6_geometry import rotation
        p = self.planner
        if p.tensor_args.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用；禁止CPU优化回退')
        solver = self.axis_solver
        arm_names = list(p.total_arm_joint_names)
        if len(arm_names) != 14 or arm_names[13] != 'arm_J7_r':
            raise ValueError('Unexpected J6 arm ordering')
        grip_values = dict(zip(p.total_gripper_joint_names, self.model.grip_matrix @ grip + self.model.grip_bias))
        # CPU FK only constructs target link coordinates; all inverse solving is CUDA IK.
        xyz, wxyz = self.model.tcp(q)
        xyz, wxyz = xyz[0].copy(), wxyz[0].copy()
        xyz[1], wxyz[1] = position, quat
        poses = {}
        for i, name in enumerate(p.target_link_names):
            flange = xyz[i]-rotation(wxyz[i]).apply(self.model.offsets[i])
            poses[name] = Pose(position=p.tensor_args.to_device(flange[None]),
                               quaternion=p.tensor_args.to_device(wxyz[i][None]))
        if list(p.target_link_names) != list(self.model.links):
            raise ValueError('GPU preparation target-link ordering mismatch')
        names = list(solver.joint_names)
        values = dict(zip(arm_names, q)); values.update(grip_values)
        seed_values = dict(zip(arm_names, seed)); seed_values.update(grip_values)
        # The fixed joints are not silently changed by the seed.
        for name in arm_names[:7]:
            seed_values[name] = values[name]
        retract = p.tensor_args.to_device([[values[name] for name in names]])
        seeds = p.tensor_args.to_device([[[seed_values[name] for name in names]]])
        began = time.monotonic()
        with preparation_bounds(solver, arm_names, self.model.bounds, q, grip_values):
            result = solver.solve_single(poses[p.ee_link_name], retract_config=retract,
                seed_config=seeds, return_seeds=solver.num_seeds, link_poses=poses)
            # Synchronize/copy before restoring shared solver tensors or releasing GPU lock.
            solutions = result.solution.detach().cpu().numpy().reshape(-1, len(names))
            success = result.success.detach().cpu().numpy().reshape(-1)
        if time.monotonic() > deadline:
            return dict(expired=True)
        candidates = []
        for row, valid in zip(solutions, success):
            if not valid or not np.isfinite(row).all():
                continue
            solved = dict(zip(names, row))
            fixed_error = max(abs(solved[name]-values[name]) for name in arm_names[:7]+list(grip_values))
            if fixed_error > 1e-5:
                continue
            candidate = q.copy()
            candidate[7:] = [solved[name] for name in arm_names[7:]]
            # Exact held joints and all geometry/collision constraints are checked again by caller.
            candidates.append(candidate.tolist())
        candidates.sort(key=lambda a: float(np.linalg.norm(np.asarray(a)[7:]-q[7:])))
        return dict(candidates=candidates, diagnostics=dict(solver='CuRobo GPU IK',
            device=str(p.tensor_args.device), num_seeds=int(solver.num_seeds),
            successful_candidates=len(candidates), gpu_success_count=int(np.count_nonzero(success)), elapsed_s=time.monotonic()-began,
            fixed_joints='left arm and both grippers',free_joints='right J1-J7', sweep_considered=False))
