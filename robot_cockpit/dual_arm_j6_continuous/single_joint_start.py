"""Bind single-wrist-joint curves to the bounded worker's measured snapshot.

Feedback drift is checked separately from commanded motion. Only the requested
J6/J7 absolute goal may move; the other thirteen joints use one measured hold.
No I/O or changes to the live Planner's state.
"""
import numpy as np

START_LIMIT_RAD = .002
EXACT_HOLD_RAD = 1e-6


def bind_arrays(incoming, measured, requested, index, move_arm):
    incoming, q, requested = [np.asarray(x, float) for x in
                              (incoming, measured, requested)]
    if (index not in (12, 13) or any(x.shape != (14,) for x in (incoming, q, requested))
            or not np.isfinite([incoming, q, requested]).all()):
        raise ValueError('Invalid single-joint start/goal')
    if len(move_arm) != 2 or bool(move_arm[0]):
        raise ValueError('Single-joint request cannot enable left arm')
    fixed = np.arange(14) != index
    input_delta = incoming-q
    held_delta = requested[fixed]-q[fixed]
    if np.max(abs(input_delta)) > START_LIMIT_RAD:
        raise ValueError('Single-joint input start changed beyond measured-start tolerance')
    if np.max(abs(held_delta)) > START_LIMIT_RAD:
        raise ValueError('Single-joint held goal changed beyond measured-start tolerance')
    target = q.copy()
    target[index] = requested[index]
    detail = dict(policy='single_measured_snapshot_preserve_moving_goal',
        moving_joint=f'right J{index-6}', fixed_joint_count=13,
        input_start_joints_rad=incoming.tolist(), measured_start_joints_rad=q.tolist(),
        requested_goal_joints_rad=requested.tolist(), effective_goal_joints_rad=target.tolist(),
        max_input_start_delta_rad=float(np.max(abs(input_delta))),
        max_held_goal_delta_rad=float(np.max(abs(held_delta))),
        start_match_limit_rad=START_LIMIT_RAD, commanded_hold_limit_rad=EXACT_HOLD_RAD)
    return q.copy(), target, detail


def bind_states(p, start, goal, index, report):
    from curobo.types.state import JointState
    names = list(p.total_arm_joint_names)
    incoming = start.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
    requested = goal.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(14)
    report.update(input_start_joints_rad=incoming.tolist(), requested_goal_joints_rad=requested.tolist(),
                  measured_start_joints_rad=np.asarray(p.arm_joint_position).tolist())
    q, target, detail = bind_arrays(incoming, p.arm_joint_position, requested, index, p._move_arm)
    report['start_binding'] = detail
    full = start.position.detach().cpu().numpy().reshape(1, -1).copy()
    full_goal = goal.get_ordered_joint_state(start.joint_names).position.detach().cpu().numpy().reshape(1, -1)
    inactive = [i for i, name in enumerate(start.joint_names) if name not in names]
    if inactive and np.max(abs(full[:, inactive]-full_goal[:, inactive])) > EXACT_HOLD_RAD:
        raise ValueError('Single-joint request cannot change grippers')
    for i, name in enumerate(names):
        full[0, start.joint_names.index(name)] = q[i]
    bound = JointState.from_position(p.tensor_args.to_device(full), joint_names=list(start.joint_names))
    return bound, q, target


def record(p, report):
    """Keep raw generator/validation evidence in this request's deadline log."""
    if hasattr(p, '_deadline_report'):
        p._deadline_report['single_joint'] = report
