"""Right-arm rehearsal geometry for approach/raise/overrun/pitch-down.

No controller imports or robot commands. User delegated provisional offsets;
all distances refer to a named contact reference, never silently to the flange.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tray_top.head import rigid_transform, select_top

DEFAULT_CONFIG = Path(__file__).with_name('right_probe.json')


def load_parameters(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text())
    for name, low, high in [('raise_clearance_mm', 0, 100), ('back_edge_overrun_mm', 0, 50),
                            ('wrist_pitch_down_deg', 0, 45), ('wrist_preview_step_deg', 0, 5)]:
        v = c[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not low < v <= high:
            raise ValueError(f'Invalid {name}: ({low}, {high}]')
    point = np.asarray(c['nominal_contact_offset_tool_mm'], float)
    if point.shape != (3,) or not np.isfinite(point).all() or np.linalg.norm(point) > 300:
        raise ValueError('Invalid tool contact reference')
    return c


def design(parameters):
    return dict(mode='right_arm_probe_design', parameters=parameters, motion_enabled=False,
                steps=['recognize the top book, approach by base Y only',
                       'fresh top-height measurement, raise vertically along base +Z',
                       'fresh same-book rear-edge measurement, advance along base +X',
                       'pitch the end effector downward, then hold for observation'],
                formulas={
                    'contact_base_mm': 'tcp_base_mm + R_base_tool @ contact_offset_tool_mm',
                    'approach_delta_y_mm': 'spine_center_y_mm - current_contact_y_mm',
                    'raise_delta_z_mm': 'top_z_mm + raise_clearance_mm - current_contact_z_mm',
                    'forward_delta_x_mm': 'observed_same_book_x_max_mm + back_edge_overrun_mm - current_contact_x_mm'},
                wrist_action='Cartesian pitch-down; joint angles are solved by CuRobo, not forced onto J7',
                j7_action='diagnostic comparison only: rotation around tool local Z',
                execution_ready=False)


def measure_top(frame, masks, head_geometry):
    mask, top, _ = select_top(frame, masks, **head_geometry)
    transform = rigid_transform(head_geometry['base_from_tray_m'])
    # Max X is computed only from depth-supported top-book inliers. It is the
    # farthest OBSERVED point, not proof of the true occluded back boundary.
    points = (top.points_m @ transform[:3, :3].T + transform[:3, 3]) * 1000.
    return dict(frame_token=frame.token, captured_unix_s=frame.captured_unix_s,
                points_base_mm=points.tolist(), sample_count=len(points),
                selected_mask_pixels=int(mask.sum()), rear_edge_observed=False,
                method='highest measured book top plane; native RGB-D; base coordinates')


def numeric_preview(observation, tcp_pose_mm_deg, parameters):
    points = np.asarray(observation['points_base_mm'], float)
    pose = np.asarray(tcp_pose_mm_deg, float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 30
            or not np.isfinite(points).all() or pose.shape != (6,) or not np.isfinite(pose).all()
            or not observation.get('frame_token')):
        raise ValueError('Need a bound RGB-D top-book point set and measured TCP pose')
    if np.linalg.matrix_rank(points[:, :2]-points[0, :2]) < 2:
        raise ValueError('Top surface has no measurable horizontal extent')
    r = Rotation.from_euler('xyz', pose[3:], degrees=True)
    offset = np.asarray(parameters['nominal_contact_offset_tool_mm'], float)
    base_offset = r.apply(offset)
    contact = pose[:3] + base_offset
    top_z = float(np.median(points[:, 2]))
    x_max = float(np.max(points[:, 0]))
    center_y = float(np.median(points[:, 1]))
    # Without a separately observed spine center, the top-footprint Y median
    # is merely a coarse candidate, not an asserted spine measurement.
    if 'spine_center_y_mm' in observation:
        center_y = float(observation['spine_center_y_mm'])
        if not np.isfinite(center_y):
            raise ValueError('Invalid measured spine center')
    align = pose[:3].copy()
    align[1] += center_y - contact[1]
    raised = align.copy()
    raised[2] = top_z + parameters['raise_clearance_mm'] - base_offset[2]
    forward = raised.copy()
    forward[0] = x_max + parameters['back_edge_overrun_mm'] - base_offset[0]
    displacements = [align - pose[:3], raised - align, forward - raised]
    if displacements[1][2] < 0 or displacements[2][0] < 0:
        raise ValueError('Requested raise/advance would go in the opposite direction; reacquire/reposition')
    stages = [dict(stage=name, tcp_target_mm=xyz.tolist(), offset_mm=delta.tolist(),
                   orientation_xyz_deg=pose[3:].tolist(), executable=False)
              for name, xyz, delta in zip(['approach_y', 'raise_z', 'advance_x'],
                                         [align, raised, forward], displacements)]

    # Downward pitch axis is horizontal and perpendicular to the measured
    # forward tool direction. This is an end-effector rotation, not J7.
    tool_forward = r.apply([0., 0., -1.])
    horizontal = tool_forward.copy()
    horizontal[2] = 0
    if np.linalg.norm(horizontal) < .5 or horizontal[0] <= 0:
        raise ValueError('Preview requires the right gripper pointing toward base +X')
    horizontal /= np.linalg.norm(horizontal)
    axis = np.cross([0., 0., 1.], horizontal)
    degrees = parameters['wrist_pitch_down_deg']
    count = int(np.ceil(degrees / parameters['wrist_preview_step_deg']))
    pitch_samples, j7_samples = [], []
    for angle in np.linspace(0., degrees, count + 1):
        pitched = Rotation.from_rotvec(axis*np.radians(angle)) * r
        j7_only = r * Rotation.from_euler('z', angle, degrees=True)
        for target, rotation in [(pitch_samples, pitched), (j7_samples, j7_only)]:
            target.append(dict(angle_deg=float(angle), tcp_mm=forward.tolist(),
                               contact_base_mm=(forward + rotation.apply(offset)).tolist(),
                               orientation_xyz_deg=rotation.as_euler('xyz', degrees=True).tolist()))
    report = design(parameters)
    report.update(mode='recorded_right_arm_probe_preview', frame_token=observation['frame_token'],
                  captured_unix_s=observation.get('captured_unix_s'),
                  measured=dict(top_z_mm=top_z, observed_x_max_mm=x_max, approach_center_y_mm=center_y,
                                center_source=('measured spine' if 'spine_center_y_mm' in observation
                                               else 'top footprint Y median; coarse candidate'),
                                x_p50_p95_max_mm=np.percentile(points[:, 0], [50, 95, 100]).tolist()),
                  stages=stages, wrist_pitch_axis_base=axis.tolist(), pitch_samples=pitch_samples,
                  j7_axis_base=r.apply([0., 0., 1.]).tolist(), j7_only_samples=j7_samples,
                  limitations=['single recorded observation is for preview only; live stages need fresh frames/feedback',
                               'observed max X is not a confirmed occluded rear edge',
                               'nominal contact point is not the complete gripper collision envelope',
                               'pitching does not establish contact force or observed book displacement',
                               'new dual-gripper collision model and CuRobo paths are not commissioned'])
    return report
