"""CuRobo 0.7 scene/held-book adapter, invoked only inside the PC Planner.

No trajectory is sent here. The owner must use the existing session-bound RT
trajectory checker and sender after planning. A scene change invalidates tickets.
"""
from dataclasses import fields, is_dataclass
import copy
import itertools
import threading
import numpy as np
from .shelf import ShelfRejected, transform

ATTACHMENT_LINK = 'book_tasks_attached_right'
GRASP_CONTACT_LINKS = ('gripper_L1_r', 'gripper_L2_r')


def sphere_box_collision(spheres, lower, upper, *, tolerance=1e-7):
    """Exact sphere/AABB intersection, accepting tangency but not penetration."""
    values = np.asarray(spheres, float)
    centers, radius = values[..., :3], values[..., 3]
    nearest = np.clip(centers, np.asarray(lower), np.asarray(upper))
    return (radius > 0) & (np.linalg.norm(centers-nearest, axis=-1) < radius-tolerance)


def obb_aabb_penetration(centers, rotations, half_sizes, lower, upper, tolerance=1e-7):
    """15-axis separating-axis test; exact rectangular book, zero contact allowed.

    `rotations` maps book-local to shelf coordinates. All samples are checked;
    there is no floor penetration allowance hidden in the sphere approximation.
    """
    centers = np.atleast_2d(centers); rotations = np.asarray(rotations)
    if rotations.ndim == 2: rotations = np.broadcast_to(rotations, (len(centers), 3, 3))
    box_center = (np.asarray(lower)+upper)*.5; box_half = (np.asarray(upper)-lower)*.5
    delta = centers-box_center; half_sizes = np.asarray(half_sizes)
    axes = [np.broadcast_to(axis, centers.shape) for axis in np.eye(3)]
    axes += [rotations[:, :, i] for i in range(3)]
    axes += [np.cross(rotations[:, :, i], axis) for i in range(3) for axis in np.eye(3)]
    penetrates = np.ones(len(centers), dtype=bool)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1); valid = norm > 1e-9
        unit = axis/np.maximum(norm[:, None], 1e-9)
        distance = np.abs(np.sum(delta*unit, axis=1))
        r_book = np.sum(np.abs(np.einsum('bij,bi->bj', rotations, unit))*half_sizes, axis=1)
        r_box = np.sum(np.abs(unit)*box_half, axis=1)
        penetrates &= (~valid) | (distance < r_book+r_box-tolerance)
    return penetrates


def covering_spheres(dimensions_m, link_to_book, capacity, uncertainty_m=.003):
    """A conservative *volume* cover, no truncation of spheres to fit a cache.

    Each regular-grid box lies entirely within its circumsphere. The grid is
    fitted to the preallocated count before construction, so every voxel stays
    represented even for long/thin books.
    """
    dims = np.asarray(dimensions_m, float)
    t = transform(link_to_book, 'attachment link-to-book')
    if dims.shape != (3,) or np.any(dims <= 0) or not np.isfinite(dims).all() or capacity < 8:
        raise ShelfRejected('Invalid attachment volume or insufficient reserved spheres')
    count = np.ones(3, dtype=int)
    while True:
        candidates = []
        for axis in range(3):
            trial = count.copy(); trial[axis] += 1
            if np.prod(trial) <= capacity:
                candidates.append((np.linalg.norm(dims/trial), axis, trial))
        if not candidates:
            break
        _, _, count = min(candidates, key=lambda x: (x[0], x[1]))
    widths = dims/count
    axes = [(np.arange(n)+.5)*width-dim*.5 for n, width, dim in zip(count, widths, dims)]
    centers = np.asarray(list(itertools.product(*axes)))
    centers = centers @ t[:3, :3].T + t[:3, 3]
    radius = float(np.linalg.norm(widths)*.5 + uncertainty_m)
    result = np.zeros((capacity, 4)); result[:, 3] = -10.
    result[:len(centers), :3] = centers; result[:len(centers), 3] = radius
    return result


class CuroboSceneSession:
    """Scoped world state; does not replace the original robot/control contract."""
    def __init__(self, planner, *, link_name=ATTACHMENT_LINK):
        self.planner = planner
        self.motion = planner._motion_gen
        self.lock = getattr(planner, '_gpu_lock', threading.RLock())
        self.link_name = link_name
        if link_name != ATTACHMENT_LINK:
            raise ShelfRejected('Never overwrite existing robot/gripper collision spheres')
        for method in ('update_world', 'attach_spheres_to_robot', 'detach_spheres_from_robot'):
            if not callable(getattr(self.motion, method, None)):
                raise ShelfRejected('Planner lacks shelf-scene capability: '+method)
        self.original_world = copy.deepcopy(self.motion.world_model)
        self.original_config_world = planner.world_cfg
        self.scene = None
        self.session = None
        self.generation = 0
        self.attached_book = None
        self.closed = False
        self.contact_books = []
        self.supports = []
        self.held_geometry = None

    def _capacity(self):
        kin = self.motion.robot_cfg.kinematics.kinematics_config
        try:
            count = kin.get_number_of_spheres(self.link_name)
        except (KeyError, ValueError, AttributeError) as error:
            raise ShelfRejected('Load the route-2 Planner profile with reserved right-book spheres') from error
        if not 8 <= count <= 512:
            raise ShelfRejected('Invalid dedicated right-book collision capacity')
        return int(count)

    def _world(self, scene, omit_names=()):
        from curobo.geom.types import Cuboid
        if not is_dataclass(self.original_world):
            raise ShelfRejected('Unsupported CuRobo world model; do not drop existing obstacles')
        kwargs = {f.name: copy.deepcopy(getattr(self.original_world, f.name))
                  for f in fields(self.original_world) if f.init and f.name != 'objects'}
        original = kwargs.get('cuboid') or []
        names = {o.name for o in getattr(self.original_world, 'objects', [])}
        additions = []
        for box in scene.obstacles:
            if box.name in omit_names or box.kind == 'support':
                continue
            name = 'book_tasks_'+box.name
            if name in names:
                raise ShelfRejected('Shelf obstacle name collides with existing world')
            values = box.as_curobo(scene.frame); values['name'] = name
            additions.append(Cuboid(**values))
        cache = self.motion.collision_cache
        if len(original)+len(additions) > int(cache.get('obb', 0)):
            raise ShelfRejected('Shelf world exceeds preallocated collision cache')
        kwargs['cuboid'] = original + additions
        return type(self.original_world)(**kwargs)

    def update(self, scene, session, *, omit_names=()):
        if self.closed or session is None or (self.session is not None and session != self.session):
            raise ShelfRejected('Shelf scene belongs to another control session')
        if self.attached_book and self.scene and self.scene.version != scene.version:
            # The caller may reobserve while holding, but must preserve the grasp
            # attachment and deliberately replan; update cannot silently detach it.
            pass
        self._capacity()
        world = self._world(scene, omit_names)
        with self.lock:
            self.motion.update_world(world)
            self.planner.world_cfg = world
            self.scene, self.session = scene, session
            self.contact_books = [box for box in scene.obstacles if box.name in omit_names]
            self.supports = [box for box in scene.obstacles if box.kind == 'support']
            self.generation += 1
            self.planner._book_tasks_scene_version = scene.version
        return self.ticket()

    def ticket(self):
        if self.scene is None or self.closed:
            raise ShelfRejected('No active shelf scene')
        return dict(scene_version=self.scene.version, generation=self.generation,
                    session=self.session, attachment=self.attached_book)

    def require(self, ticket):
        if ticket != self.ticket():
            raise ShelfRejected('Scene/session/attachment changed; old plan is invalid')

    def attach(self, geometry, book_id, link_to_tcp, ticket):
        self.require(ticket)
        if self.attached_book is not None or not book_id:
            raise ShelfRejected('A book is already attached or its identity is missing')
        link_to_book = transform(link_to_tcp, 'right link-to-TCP') @ transform(geometry.tcp_to_book)
        spheres = covering_spheres(geometry.book_dimensions_m, link_to_book, self._capacity())
        with self.lock:
            tensor = self.planner.tensor_args.to_device(spheres)
            self.motion.attach_spheres_to_robot(sphere_tensor=tensor, link_name=self.link_name)
            self.attached_book = book_id
            self.held_geometry = geometry
            self.generation += 1
        return self.ticket()

    def detach(self, book_id, ticket, *, released):
        self.require(ticket)
        if not released or book_id != self.attached_book:
            raise ShelfRejected('Cannot detach collision book without matching release evidence')
        with self.lock:
            self.motion.detach_spheres_from_robot(link_name=self.link_name)
            self.attached_book = None
            self.held_geometry = None
            self.generation += 1
        return self.ticket()

    def check_samples(self, arm_joint_rad, adapter, checkpoint, ticket):
        """Use the SAME MotionGen checker as planning on full RT interpolation."""
        self.require(ticket)
        q = np.asarray(arm_joint_rad, float)
        if q.ndim != 2 or q.shape[1] != 14 or not np.isfinite(q).all():
            raise ShelfRejected('Invalid RT trajectory samples')
        values = adapter.arm_values(q)
        for index in range(0, len(values), 512):
            checkpoint(); self.require(ticket)
            with self.lock:
                result = self.motion.rollout_fn.rollout_constraint(
                    self.planner.tensor_args.to_device(np.ascontiguousarray(values[index:index+512])).unsqueeze(1),
                    use_batch_env=False)
                if not bool(result.feasible.all().item()):
                    raise ShelfRejected('Shelf/held-book collision or joint constraint on RT path')
                if self.supports or (self.contact_books and self.attached_book is None):
                    # A selected grasp target may contact the right fingers,
                    # but remains an obstacle to every other robot link.
                    kin = self.motion.kinematics
                    state = kin.get_state(self.planner.tensor_args.to_device(
                        np.ascontiguousarray(values[index:index+512])))
                    spheres = state.link_spheres_tensor.detach().cpu().numpy().copy()
                    spheres[..., :3] = self.scene.frame.from_base(spheres[..., :3])
                    attachment_indices = kin.kinematics_config.get_sphere_index_from_link_name(self.link_name).detach().cpu().numpy()
                    # Robot/finger floor protection stays intact. Only the held
                    # book uses its measured cuboid instead of over-covering balls.
                    spheres[:, attachment_indices, 3] = -10.
                    for box in self.supports:
                        if sphere_box_collision(spheres, box.lower, box.upper).any():
                            raise ShelfRejected('Robot/gripper penetrates a shelf support')
                    if self.attached_book is None:
                        for link in GRASP_CONTACT_LINKS:
                            indices = kin.kinematics_config.get_sphere_index_from_link_name(link).detach().cpu().numpy()
                            spheres[:, indices, 3] = -10.
                        for box in self.contact_books:
                            if sphere_box_collision(spheres, box.lower, box.upper).any():
                                raise ShelfRejected('Selected book contacts a non-grasp robot link')
                    elif self.supports:
                        from scipy.spatial.transform import Rotation
                        positions, quaternions = adapter.tcp(q[index:index+512])
                        right_rotation = Rotation.from_quat(quaternions[:, 1, [1, 2, 3, 0]]).as_matrix()
                        offset = np.asarray(self.held_geometry.tcp_to_book)
                        centers = positions[:, 1]+np.einsum('bij,j->bi', right_rotation, offset[:3, 3])
                        frame_rotation = np.asarray(self.scene.frame.shelf_to_base)[:3, :3]
                        rotations = np.einsum('ij,bjk,kl->bil', frame_rotation.T, right_rotation, offset[:3, :3])
                        centers = self.scene.frame.from_base(centers)
                        for box in self.supports:
                            if obb_aabb_penetration(centers, rotations,
                                    np.asarray(self.held_geometry.book_dimensions_m)*.5, box.lower, box.upper).any():
                                raise ShelfRejected('Held book penetrates measured shelf support')
        checkpoint(); self.require(ticket)

    def close(self, *, robot_stopped):
        if not robot_stopped:
            raise ShelfRejected('Cannot change collision world during active execution')
        if self.attached_book is not None:
            raise ShelfRejected('Keep the held-book collision model until release is confirmed')
        with self.lock:
            self.motion.update_world(self.original_world)
            self.planner.world_cfg = self.original_config_world
            self.planner._book_tasks_scene_version = None
            self.closed = True
            self.generation += 1
