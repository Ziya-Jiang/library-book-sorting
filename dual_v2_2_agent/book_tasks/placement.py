"""Reserve the fixed right neighbour and generate measured shelf waypoints.

Waypoints are NOT authorization to move: every segment must pass the Planner
scene/attached-object transaction, including the post-release retreat.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .shelf import Box, ShelfRejected, transform, vector, canonical_digest


def box_corners(lower, upper):
    return np.asarray([[x, y, z] for x in (lower[0], upper[0])
                       for y in (lower[1], upper[1]) for z in (lower[2], upper[2])], float)


@dataclass(frozen=True)
class HeldGeometry:
    """Book axes are (thickness/right, depth/inward, height/up).

    tcp_to_book maps book-local points to the actual right TCP frame. Gripper
    bounds are the open-gripper envelope in that TCP frame, not the closing gap.
    """
    book_dimensions_m: list
    tcp_to_book: list
    open_gripper_lower_m: list
    open_gripper_upper_m: list
    evidence: str

    def __post_init__(self):
        dims = vector(self.book_dimensions_m, 3, 'measured book dimensions')
        transform(self.tcp_to_book, 'TCP-to-book')
        Box('open_gripper', self.open_gripper_lower_m, self.open_gripper_upper_m)
        if np.any(dims < .002) or np.any(dims > .8) or not self.evidence:
            raise ShelfRejected('Book dimensions and grasp transform need measurement evidence')

    def tcp_pose(self, scene, book_center):
        book = np.asarray(scene.frame.shelf_to_base, float).copy()
        book[:3, 3] = scene.frame.to_base(book_center)
        return book @ np.linalg.inv(np.asarray(self.tcp_to_book))

    def envelope(self, scene, tcp_pose):
        corners = box_corners(self.open_gripper_lower_m, self.open_gripper_upper_m)
        base = corners @ tcp_pose[:3, :3].T + tcp_pose[:3, 3]
        local = scene.frame.from_base(base)
        return Box('open_gripper', local.min(axis=0).tolist(), local.max(axis=0).tolist(), 'gripper')


@dataclass(frozen=True)
class PlacementPlan:
    scene_version: str
    source_cell: str
    target_cell: str
    book_id: str
    placed_book: dict
    open_gripper: dict
    waypoints: tuple
    geometry_evidence: str
    plan_id: str = ''

    def __post_init__(self):
        data = asdict(self)
        data.pop('plan_id')
        digest = canonical_digest(data)
        if self.plan_id and self.plan_id != digest:
            raise ShelfRejected('Placement plan digest mismatch')
        object.__setattr__(self, 'plan_id', digest)

    def to_dict(self):
        return asdict(self)

    def require_scene(self, scene):
        if self.scene_version != scene.version:
            raise ShelfRejected('Shelf scene changed; discard the old placement plan')


def reserve_placement(scene, binding, geometry, book_id, *, clearance_m=.01,
                      transfer_clearance_m=.05, placement_lift_m=.015):
    """Find the rightmost collision-free placement in the fixed destination.

    The first placement starts left of the rightmost measured bookend. Later
    placements progress leftward; it never fills a hole to the right of an
    already placed book or silently chooses another shelf compartment.
    """
    source, target = binding.validate(scene)
    if not book_id or not .005 <= clearance_m <= .05:
        raise ShelfRejected('Invalid placement identity or clearance')
    if not target.bookends:
        raise ShelfRejected('Fixed bookend is no longer observed')
    dims = np.asarray(geometry.book_dimensions_m, float)
    boundary = max(Box(**b).lower[0] for b in target.bookends)
    if target.occupied:
        boundary = min(boundary, min(Box(**b).lower[0] for b in target.occupied))
    obstacles = [Box(**b) for b in target.bookends + target.occupied]
    # Candidate spacing follows the actual open-gripper footprint; no dense insertion.
    v = target.lower[1] + dims[1]*.5 + clearance_m
    w = target.lower[2] + dims[2]*.5
    nominal = np.asarray([boundary-dims[0]*.5-clearance_m, v, w])
    tcp = geometry.tcp_pose(scene, nominal)
    envelope = geometry.envelope(scene, tcp)
    nominal[0] -= max(0., envelope.upper[0] - (boundary-clearance_m))
    tcp = geometry.tcp_pose(scene, nominal)
    envelope = geometry.envelope(scene, tcp)
    book = Box(book_id, (nominal-dims*.5).tolist(), (nominal+dims*.5).tolist(), 'book')
    if not target.contains(book):
        raise ShelfRejected('Right neighbour has insufficient book space')
    # Fingers may remain outside the front opening, but not behind/above/below/sideways.
    for axis in (0, 2):
        if (envelope.lower[axis] < target.lower[axis]+clearance_m
                or envelope.upper[axis] > target.upper[axis]-clearance_m):
            raise ShelfRejected('Open gripper cannot enter or withdraw through target opening')
    if envelope.upper[1] > target.upper[1]-clearance_m:
        raise ShelfRejected('Open gripper would hit the shelf back wall')
    if any(book.intersects(b, clearance_m) or envelope.intersects(b, clearance_m) for b in obstacles):
        raise ShelfRejected('Book or open gripper overlaps existing occupancy/bookend')
    if book.upper[2]+placement_lift_m+clearance_m > target.upper[2]:
        raise ShelfRejected('Insufficient ceiling clearance for lowering onto support')
    # Floor contact is a geometric goal only; no collision filter is disabled for it.
    raised = nominal.copy(); raised[2] += placement_lift_m
    pre = raised.copy(); pre[1] = target.lower[1] - dims[1]*.5 - transfer_clearance_m
    retreat = nominal.copy(); retreat[1] = pre[1]
    stages = [('target_prepose', pre, 'held'), ('insert', raised, 'held'),
              ('lower_to_support', nominal, 'held'), ('retreat', retreat, 'released')]
    waypoints = tuple(dict(stage=name, tcp_pose_base=geometry.tcp_pose(scene, center).tolist(),
                           book_center_shelf=center.tolist(), attachment=attachment)
                      for name, center, attachment in stages)
    return PlacementPlan(scene.version, source.cell_id, target.cell_id, book_id,
                         asdict(book), asdict(envelope), waypoints, geometry.evidence)


def extraction_waypoints(scene, binding, geometry, current_tcp_pose, *, clearance_m=.05,
                         lift_m=.015):
    """Constrained source lift/extraction before free-space transfer, all metres."""
    source, _ = binding.validate(scene)
    current = transform(current_tcp_pose, 'measured right TCP')
    book_t = current @ np.asarray(geometry.tcp_to_book)
    local = scene.frame.from_base(book_t[:3, 3])
    dims = np.asarray(geometry.book_dimensions_m)
    if local[2]+dims[2]*.5+lift_m > source.upper[2]-.01:
        raise ShelfRejected('Source ceiling blocks extraction lift')
    raised = local.copy(); raised[2] += lift_m
    front = raised.copy(); front[1] = source.lower[1]-dims[1]*.5-clearance_m
    return [dict(stage=name, tcp_pose_base=geometry.tcp_pose(scene, center).tolist(),
                 book_center_shelf=center.tolist(), attachment='held')
            for name, center in [('source_lift', raised), ('source_extract', front)]]


def confirm_placement(plan, observations, *, released, position_tolerance_m=.015,
                      stationary_tolerance_m=.005, minimum_interval_s=.5):
    """Require two independent post-release metric observations of the same book.

    Caller provides associated observed book boxes, not commanded TCP poses.
    A scene without detections is not proof of successful placement.
    """
    if not released or len(observations) < 2:
        raise ShelfRejected('Need release evidence and two post-release book observations')
    a, b = observations[-2:]
    if a['observation_id'] == b['observation_id'] or b['captured_unix_s']-a['captured_unix_s'] < minimum_interval_s:
        raise ShelfRejected('Placement observations are repeated or too close in time')
    expected = Box(**plan.placed_book)
    boxes = []
    for item in (a, b):
        if item['book_id'] != plan.book_id or item['cell_id'] != plan.target_cell:
            raise ShelfRejected('Placed book was not associated with the reserved destination')
        if not item.get('support_confirmed') or not item.get('upright_confirmed'):
            raise ShelfRejected('Placed book lacks observed support/upright evidence')
        box = Box(**item['box'])
        if np.linalg.norm(box.center-expected.center) > position_tolerance_m:
            raise ShelfRejected('Placed book lies outside expected position')
        boxes.append(box)
    if np.linalg.norm(boxes[1].center-boxes[0].center) > stationary_tolerance_m:
        raise ShelfRejected('Placed book is still moving or falling')
    return dict(book_id=plan.book_id, target_cell=plan.target_cell,
                verified_observations=[a['observation_id'], b['observation_id']], stable=True)
