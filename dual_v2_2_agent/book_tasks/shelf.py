"""Measured shelf geometry. All distances are metres, rotations are radians.

Shelf coordinates are (u=operator's right, v=into the shelf, w=up).
No camera, model, planner or robot is opened by this module.
"""
from dataclasses import dataclass, field, asdict, replace
import hashlib
import json
import math
import numpy as np


class ShelfRejected(ValueError):
    pass


def vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ShelfRejected('Invalid '+label)
    return result


def transform(value, label='transform'):
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-6)):
        raise ShelfRejected('Invalid '+label)
    return result


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Frame:
    """Rigid shelf-to-base transform, explicit units (m)."""
    shelf_to_base: list

    def __post_init__(self):
        transform(self.shelf_to_base, 'shelf_to_base')

    def to_base(self, points):
        points = np.asarray(points, float)
        t = np.asarray(self.shelf_to_base)
        return points @ t[:3, :3].T + t[:3, 3]

    def from_base(self, points):
        points = np.asarray(points, float)
        t = np.asarray(self.shelf_to_base)
        return (points - t[:3, 3]) @ t[:3, :3]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in the named shelf frame, never a guessed pixel depth."""
    name: str
    lower: list
    upper: list
    kind: str = 'obstacle'

    def __post_init__(self):
        lower = vector(self.lower, 3, 'box lower')
        upper = vector(self.upper, 3, 'box upper')
        if not self.name or np.any(upper <= lower):
            raise ShelfRejected('Empty or unnamed box')

    @property
    def center(self):
        return (np.asarray(self.lower) + np.asarray(self.upper)) * .5

    @property
    def dimensions(self):
        return np.asarray(self.upper) - self.lower

    def intersects(self, other, margin=0.):
        return bool(np.all(np.asarray(self.lower) - margin < np.asarray(other.upper))
                    and np.all(np.asarray(other.lower) - margin < np.asarray(self.upper)))

    def as_curobo(self, frame):
        from scipy.spatial.transform import Rotation
        t = np.asarray(frame.shelf_to_base)
        xyzw = Rotation.from_matrix(t[:3, :3]).as_quat()
        return dict(name=self.name, pose=[*frame.to_base(self.center).tolist(),
                    float(xyzw[3]), *xyzw[:3].tolist()], dims=self.dimensions.tolist())


@dataclass(frozen=True)
class Cell:
    cell_id: str
    row: int
    column: int
    lower: list
    upper: list
    bookends: list = field(default_factory=list)
    occupied: list = field(default_factory=list)
    measured: bool = True

    def __post_init__(self):
        Box(self.cell_id, self.lower, self.upper)
        if type(self.row) is not int or type(self.column) is not int or not self.measured:
            raise ShelfRejected('Cell needs measured geometry and integer row/column')
        for name in ('bookends', 'occupied'):
            for item in getattr(self, name):
                Box(**item)

    def contains(self, box, margin=0.):
        return bool(np.all(np.asarray(box.lower) >= np.asarray(self.lower) + margin)
                    and np.all(np.asarray(box.upper) <= np.asarray(self.upper) - margin))


@dataclass(frozen=True)
class ShelfScene:
    observation_id: str
    captured_unix_s: float
    frame: Frame
    cells: tuple
    obstacles: tuple = ()
    geometry_complete: bool = True
    version: str = ''

    def __post_init__(self):
        if not self.observation_id or not math.isfinite(self.captured_unix_s) or not self.geometry_complete:
            raise ShelfRejected('Shelf observation is incomplete')
        if len(self.cells) < 2 or len({c.cell_id for c in self.cells}) != len(self.cells):
            raise ShelfRejected('Need uniquely identified source and neighbour cells')
        if len({(c.row, c.column) for c in self.cells}) != len(self.cells):
            raise ShelfRejected('Ambiguous shelf grid')
        body = self.to_dict(include_version=False)
        expected = canonical_digest(body)
        if self.version and self.version != expected:
            raise ShelfRejected('Shelf scene digest mismatch')
        object.__setattr__(self, 'version', expected)

    def to_dict(self, include_version=True):
        data = dict(observation_id=self.observation_id, captured_unix_s=self.captured_unix_s,
                    frame=asdict(self.frame), cells=[asdict(c) for c in self.cells],
                    obstacles=[asdict(b) for b in self.obstacles], geometry_complete=self.geometry_complete)
        if include_version:
            data['version'] = self.version
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(observation_id=data['observation_id'], captured_unix_s=data['captured_unix_s'],
                   frame=Frame(**data['frame']), cells=tuple(Cell(**c) for c in data['cells']),
                   obstacles=tuple(Box(**b) for b in data.get('obstacles', [])),
                   geometry_complete=data.get('geometry_complete', False), version=data.get('version', ''))

    def cell(self, cell_id):
        matches = [c for c in self.cells if c.cell_id == cell_id]
        if len(matches) != 1:
            raise ShelfRejected('Unknown or ambiguous shelf cell: '+str(cell_id))
        return matches[0]

    def right_neighbour(self, source_id):
        source = self.cell(source_id)
        candidates = [c for c in self.cells if c.row == source.row and c.column == source.column + 1]
        if len(candidates) != 1:
            raise ShelfRejected('The immediate same-row right neighbour was not measured')
        target = candidates[0]
        # Do not accept a second detector's reversed or vertically mislabelled grid.
        if (target.lower[0] < source.upper[0] - .005
                or abs(target.lower[2] - source.lower[2]) > .015
                or abs(target.lower[1] - source.lower[1]) > .03):
            raise ShelfRejected('Geometric same-row adjacency disagrees with cell IDs')
        if target.lower[0] - source.upper[0] > .10:
            raise ShelfRejected('Unobserved gap between source and purported neighbour')
        return target


@dataclass
class CellBinding:
    """Human confirmation locks source/target IDs and their physical geometry."""
    source_id: str
    target_id: str
    source_lower: list
    source_upper: list
    target_lower: list
    target_upper: list
    initial_scene_version: str
    frame: list

    @classmethod
    def confirm(cls, scene, source_id):
        source = scene.cell(source_id)
        target = scene.right_neighbour(source_id)
        if target.occupied:
            raise ShelfRejected('First shelf batch requires an initially empty target cell')
        if not target.bookends:
            raise ShelfRejected('A measured fixed bookend is required for the first book')
        return cls(source_id, target.cell_id, source.lower, source.upper, target.lower,
                   target.upper, scene.version, scene.frame.shelf_to_base)

    def validate(self, scene, tolerance=.01):
        source = scene.cell(self.source_id)
        target = scene.right_neighbour(self.source_id)
        if target.cell_id != self.target_id:
            raise ShelfRejected('Target neighbour changed')
        old_frame = Frame(self.frame)
        for old_lower, old_upper, current in [(self.source_lower, self.source_upper, source),
                                               (self.target_lower, self.target_upper, target)]:
            def corners(lower, upper):
                return [[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
            old = old_frame.to_base(corners(old_lower, old_upper))
            new = scene.frame.to_base(corners(current.lower, current.upper))
            if np.max(np.linalg.norm(old-new, axis=1)) > tolerance:
                raise ShelfRejected('Shelf moved or cell association changed; confirm again')
        return source, target


def selected_book_scene(scene, book_id, static_name, measured_box):
    """Replace one semantic occupied column with its associated measured book.

    Returns a new version. Other books and shelf obstacles remain conservative.
    The returned book's name can be passed to update_scene(omit_names=[book_id]);
    Planner then checks it against every non-grasp robot link at every RT sample.
    """
    if not book_id or measured_box.kind != 'book':
        raise ShelfRejected('Selected grasp target needs measured book geometry')
    matches = [box for box in scene.obstacles if box.name == static_name and box.kind == 'book']
    if len(matches) != 1:
        raise ShelfRejected('Selected book has no unique observed occupancy association')
    old = matches[0]
    if (np.any(np.asarray(measured_box.lower) < np.asarray(old.lower)-.015)
            or np.any(np.asarray(measured_box.upper) > np.asarray(old.upper)+.015)):
        raise ShelfRejected('Measured target disagrees with its observed book column')
    precise = replace(measured_box, name=book_id)
    cells = []
    associated = 0
    for cell in scene.cells:
        occupied = []
        for box in cell.occupied:
            if box['name'] == static_name:
                if not cell.contains(precise): raise ShelfRejected('Measured book lies outside source cell')
                occupied.append(asdict(precise)); associated += 1
            else: occupied.append(box)
        cells.append(replace(cell, occupied=occupied))
    if associated != 1: raise ShelfRejected('Selected occupancy is not uniquely assigned to a cell')
    obstacles = tuple(precise if box.name == static_name else box for box in scene.obstacles)
    return replace(scene, cells=tuple(cells), obstacles=obstacles, version='')
