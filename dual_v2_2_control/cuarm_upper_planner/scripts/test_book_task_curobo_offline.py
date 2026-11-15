"""MANUAL GPU contract verification, without a Planner, socket or robot process.

Run with the existing CuRobo environment and the same --config-root as the live
Planner. This allocates a separate MotionGen but does not plan/send any motion.
It is intentionally outside automatic unittest discovery (no TestCase).
"""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from types import SimpleNamespace


def reject_sockets(event, args):
    if event in ('socket.__new__', 'socket.bind', 'socket.connect', 'socket.sendto'):
        raise RuntimeError('Offline task contract test forbids every socket operation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', default=os.environ.get('CUARM_CONFIG_ROOT', '/mnt/ssd/user/V/dual_v2_2_configuration'))
    parser.add_argument('--minimum-free-gib', type=float, default=4.)
    parser.add_argument('--report')
    parser.add_argument('--plan', action='store_true', help='Also solve a small synthetic right-arm goal and validate its complete RT interpolation')
    args = parser.parse_args(argv)
    sys.addaudithook(reject_sockets)
    root = Path(args.config_root).resolve()
    arch = 'arm64' if platform.machine().lower() in ('aarch64', 'arm64') else 'x86_64'
    sys.path[:0] = [str(root/'robot_platform_utils/python/scripts'),
                   str(root/'robot_platform_utils/python/lib'/arch),
                   str(root/'robot_platform_utils/python/lib/x86'), str(root/'curi_udp/python')]
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from curobo_config import CuroboConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.wrap.reacher.evaluator import TrajEvaluatorConfig
    from book_task_planner import isolated_motion_gen
    from book_tasks.planner_scene import CuroboSceneSession, ATTACHMENT_LINK, obb_aabb_penetration
    from book_tasks.placement import HeldGeometry
    from book_tasks.shelf import Frame, Box, Cell, ShelfScene

    if not torch.cuda.is_available(): raise RuntimeError('Use the actual CUDA CuRobo environment')
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib*1024**3:
        raise RuntimeError(f'Only {free/1024**3:.2f} GiB GPU free; offline task engine was not constructed')
    config = CuroboConfig.load_from_yaml_path(str(root/'config.yaml'))
    tensor_args = TensorDeviceType()
    original_cfg = copy.deepcopy(config.robot_cfg)
    original_robot = RobotConfig.from_dict(copy.deepcopy(original_cfg), tensor_args)
    original_kin = CudaRobotModel(original_robot.kinematics)
    options = dict(trajopt_tsteps=10, collision_checker_type=CollisionCheckerType.MESH,
        collision_activation_distance=0., ik_opt_iters=200, use_cuda_graph=True, ik_particle_opt=True,
        num_ik_seeds=12, num_trajopt_seeds=12, num_graph_seeds=1, interpolation_dt=.02,
        self_collision_check=True, self_collision_opt=True, maximum_trajectory_time=7.,
        minimum_trajectory_dt=.2, trim_steps=[2, -2], position_threshold=.001,
        rotation_threshold=.1, cspace_threshold=.05,
        traj_evaluator_config=TrajEvaluatorConfig.from_basic(min_dt=.2, max_dt=.7,
            dof=original_robot.kinematics.dof))
    planner = SimpleNamespace(robot_cfg=copy.deepcopy(original_cfg), tensor_args=tensor_args,
        target_link_names=list(config.target_link_names), world_cfg=copy.deepcopy(config.world_cfg),
        _gpu_lock=threading.RLock(), _book_tasks_motion_options=options)
    motion = isolated_motion_gen(planner); planner._motion_gen = motion
    assert json.dumps(planner.robot_cfg, sort_keys=True) == json.dumps(original_cfg, sort_keys=True)
    kin = motion.robot_cfg.kinematics.kinematics_config
    assert kin.get_number_of_spheres(ATTACHMENT_LINK) == 128
    assert motion.collision_cache['obb'] >= 256
    q = tensor_args.to_device(original_cfg['kinematics']['cspace']['retract_config']).reshape(1, -1)
    before = original_kin.get_state(q)
    original_positions = before.links_position.detach().cpu().numpy().copy()
    original_rotations = before.links_quaternion.detach().cpu().numpy().copy()
    after = motion.kinematics.get_state(q)
    for link in config.target_link_names:
        left = before.link_names.index(link); right = after.link_names.index(link)
        np.testing.assert_allclose(original_positions[:, left], after.links_position[:, right].detach().cpu().numpy(), atol=1e-6)
        dot = np.sum(original_rotations[:, left]*after.links_quaternion[:, right].detach().cpu().numpy(), axis=1)
        np.testing.assert_allclose(np.abs(dot), [1.], atol=1e-5)
    right_index = after.link_names.index('arm_end_effector_r')
    link_position = after.links_position[0, right_index].detach().cpu().numpy().copy()
    link_quaternion = after.links_quaternion[0, right_index].detach().cpu().numpy().copy()
    link_rotation = Rotation.from_quat(link_quaternion[[1, 2, 3, 0]]).as_matrix()
    world_before = copy.deepcopy(motion.world_model)
    # Test obstacles are far from the robot; no real scene/robot state is read.
    frame = Frame(np.eye(4).tolist())
    cells = (Cell('a', 0, 0, [3., 0., 0.], [3.5, .4, .5]),
             Cell('b', 0, 1, [3.52, 0., 0.], [4.02, .4, .5]))
    floor = Box('offline_floor', [3., 0., -.02], [4.02, .4, 0.], 'support')
    wall = Box('offline_wall', [4.02, 0., 0.], [4.04, .4, .5], 'shelf')
    scene = ShelfScene('offline_only', 1., frame, cells, (floor, wall))
    session = CuroboSceneSession(planner)
    first = session.update(scene, 'offline_no_robot_session')
    names = {item.name for item in motion.world_model.objects}
    assert 'book_tasks_offline_wall' in names and 'book_tasks_offline_floor' not in names
    assert session.supports == [floor]
    tcp = np.eye(4); tcp[:3, 3] = config.end_effector_offset[1]
    grip_book = np.eye(4); grip_book[:3, 3] = [.035, 0., -.02]
    geometry = HeldGeometry([.04, .2, .3], grip_book.tolist(), [-.03, -.04, -.04], [.03, .04, .04], 'offline fixture')
    ticket = session.attach(geometry, 'offline_book', tcp, first)
    local_spheres = kin.get_link_spheres(ATTACHMENT_LINK).detach().cpu().numpy().copy()
    attached_state = motion.kinematics.get_state(q)
    indices = kin.get_sphere_index_from_link_name(ATTACHMENT_LINK).detach().cpu().numpy()
    world_spheres = attached_state.link_spheres_tensor[0, indices].detach().cpu().numpy()
    active = local_spheres[:, 3] > 0
    expected = local_spheres[active, :3]@link_rotation.T+link_position
    np.testing.assert_allclose(world_spheres[active, :3], expected, atol=2e-6)
    np.testing.assert_allclose(local_spheres[active, :3].mean(axis=0), (tcp@grip_book)[:3, 3], atol=1e-6)
    assert np.all(world_spheres[active, 3] > 0)
    old_ticket = ticket
    refreshed = replace(scene, observation_id='offline_fresh_frame', version='')
    ticket = session.update(refreshed, 'offline_no_robot_session')
    assert session.attached_book == 'offline_book'
    np.testing.assert_allclose(kin.get_link_spheres(ATTACHMENT_LINK).detach().cpu().numpy(), local_spheres)
    try: session.require(old_ticket)
    except ValueError: pass
    else: raise AssertionError('A scene update failed to invalidate the previous trajectory generation')
    session.detach('offline_book', ticket, released=True)
    assert np.all(kin.get_link_spheres(ATTACHMENT_LINK).detach().cpu().numpy()[:, 3] < 0)

    # Real CUDA constraint positive/negative cases, using no measured robot I/O.
    # Search a deterministic neighbourhood of the configured retract state if
    # that exact state is obstructed by the original static collision world.
    all_names = list(config.joint_names)
    arm_indices = [all_names.index(name) for name in config.total_arm_joint_names]
    grip_indices = [all_names.index(name) for name in config.total_gripper_joint_names]
    limits = original_robot.kinematics.get_joint_limits().position.detach().cpu().numpy()
    reference = q.detach().cpu().numpy()[0]
    rng = np.random.default_rng(4311)
    candidates = np.repeat(reference[None], 1025, axis=0)
    candidates[1:, arm_indices] += rng.normal(0., .35, (1024, len(arm_indices)))
    candidates[:, arm_indices] = np.clip(candidates[:, arm_indices],
        limits[0, arm_indices]*.99, limits[1, arm_indices]*.99)
    with planner._gpu_lock:
        feasible = motion.rollout_fn.rollout_constraint(tensor_args.to_device(candidates).unsqueeze(1),
            use_batch_env=False).feasible.detach().cpu().numpy().reshape(len(candidates), -1).all(axis=1)
    if not feasible.any():
        raise RuntimeError('No synthetic collision-free retract-neighbour state; no motion planner was called')
    reference = candidates[int(np.flatnonzero(feasible)[0])].copy()
    arm_state = reference[arm_indices]
    def arm_values(values):
        values = np.atleast_2d(values)
        result = np.repeat(reference[None], len(values), axis=0)
        result[:, arm_indices] = values
        return result
    def tcp_values(values):
        state = original_kin.get_state(tensor_args.to_device(arm_values(values)))
        indices = [state.link_names.index(name) for name in config.target_link_names]
        positions = state.links_position[:, indices].detach().cpu().numpy().copy()
        orientations = state.links_quaternion[:, indices].detach().cpu().numpy().copy()
        for i in range(2):
            positions[:, i] += Rotation.from_quat(orientations[:, i, [1, 2, 3, 0]]).apply(config.end_effector_offset[i])
        return positions, orientations
    adapter = SimpleNamespace(arm_values=arm_values, tcp=tcp_values)
    session.check_samples(np.repeat(arm_state[None], 3, axis=0), adapter, lambda: None, session.ticket())
    state = motion.kinematics.get_state(tensor_args.to_device(reference).reshape(1, -1))
    spheres = state.link_spheres_tensor.detach().cpu().numpy()[0].copy()
    active_sphere = spheres[np.flatnonzero(spheres[:, 3] > .03)[0]]
    center = active_sphere[:3]; collision_floor = Box('reject_floor', (center-.01).tolist(), (center+.01).tolist(), 'support')
    blocked = replace(scene, obstacles=scene.obstacles+(collision_floor,), version='')
    session.update(blocked, 'offline_no_robot_session')
    try:
        session.check_samples(arm_state[None], adapter, lambda: None, session.ticket())
    except ValueError as error:
        assert 'shelf support' in str(error), str(error)
    else: raise AssertionError('Exact robot/floor check failed to reject penetration')
    contact_book = replace(collision_floor, name='reject_contact', kind='book')
    blocked = replace(scene, obstacles=scene.obstacles+(contact_book,), version='')
    session.update(blocked, 'offline_no_robot_session', omit_names=[contact_book.name])
    try:
        session.check_samples(arm_state[None], adapter, lambda: None, session.ticket())
    except ValueError as error:
        assert 'non-grasp robot link' in str(error), str(error)
    else: raise AssertionError('Selected target was incorrectly ignored for non-gripper links')
    session.update(scene, 'offline_no_robot_session')

    motion_plan_report = dict(requested=False)
    if args.plan:
        from baseline_approach_planner import Adapter
        from book_task_planner import Transactions
        from v3_1_panel_path import PanelPath
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        planner.config = config.config
        planner.arm_soft_joint_limits = limits[:, arm_indices]*.99
        planner.target_gripper_joint = reference[grip_indices].tolist()
        planner.total_arm_joint_names = list(config.total_arm_joint_names)
        planner.curobo_input_joint_names = all_names
        planner.end_effector_offset = config.end_effector_offset
        planner.ee_link_idx_in_target_link_names = config.ee_link_idx_in_target_link_names
        planner._baseline_speed_factor = None
        planner._plan_config = MotionGenPlanConfig(check_start_validity=True, time_dilation_factor=.4)
        planner.reorder_input = lambda arms, grippers, as_curobo: [
            (list(arms)+list(grippers))[(config.total_arm_joint_names+config.total_gripper_joint_names).index(name)]
            for name in all_names]
        configured_velocity = np.deg2rad([arm['safety']['joint_soft_limit']['velocity']
                                         for arm in config.config['robot']['arm']])
        planner._v31_bridge = SimpleNamespace(tcp=tcp_values, panel_path=PanelPath(),
            limits=configured_velocity*.1, joint_velocity_limits=configured_velocity.copy())
        actual_adapter = Adapter(planner)
        actual_adapter.check = lambda: None  # Offline only: no panel/session exists or is created.
        transaction = Transactions(planner); transaction.scene = session
        goal_joints = arm_state.copy(); goal_joints[7] += .005
        positions, orientations = tcp_values(goal_joints)
        target = np.eye(4); target[:3, 3] = positions[0, 1]
        target[:3, :3] = Rotation.from_quat(orientations[0, 1, [1, 2, 3, 0]]).as_matrix()
        path = transaction.plan(actual_adapter, arm_state, target, linear=False, deadline=time.monotonic()+60.)
        motion_plan_report = dict(requested=True, success=True, trajectory_waypoints=len(path['points']),
                                  rt_validation=True, duration_s=path['duration_s'], executable=False)
    session.close(robot_stopped=True)
    assert {x.name for x in motion.world_model.objects} == {x.name for x in world_before.objects}
    np.testing.assert_array_equal(obb_aabb_penetration([[3.2, .2, .15], [3.2, .2, .149]],
        np.eye(3), [.02, .1, .15], floor.lower, floor.upper), [False, True])
    report = dict(offline=True, robot_commands_sent=0, sockets_created=0, model_api_calls=0,
        isolated_motion_gen=True, attachment_spheres=128, original_fk_unchanged=True,
        right_link_tcp_attachment_verified=True, world_update_restore=True,
        attached_scene_update_preserves_book=True, changed_scene_invalidates_ticket=True,
        exact_book_support_contact=True, gpu_free_before_gib=free/1024**3,
        real_cuda_rt_check_positive=True, real_cuda_robot_support_rejection=True,
        real_cuda_non_gripper_contact_rejection=True, motion_plan=motion_plan_report,
        gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2,
        note=('Offline synthetic motion plan only, no execution or field acceptance' if args.plan else
              'Real CUDA configuration/FK/world/attachment contract only; no motion plan or field acceptance'))
    if args.report: Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
