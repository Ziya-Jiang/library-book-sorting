"""Read-only stage handoff checks; no target rebasing or control acquisition."""
import json
import time
import numpy as np

JOINT_START_LIMIT_RAD = float(np.deg2rad(0.5))
SETTLED_DRIFT_RAD = .001


def check_start(evidence, state, read_arrays):
    joints, _, grip = read_arrays(state)
    delta = joints - np.asarray(evidence['start_joints_rad'], float)
    index = int(np.argmax(np.abs(delta)))
    causes = []
    if state.get('epoch') != evidence['epoch']:
        causes.append('session_epoch_changed')
    if abs(delta[index]) > JOINT_START_LIMIT_RAD:
        causes.append('joint_start_changed')
    if not np.array_equal(grip, evidence['gripper_feedback']):
        causes.append('gripper_echo_changed')
    if not causes:
        return
    detail = dict(causes=causes, unix_s=time.time(),
                  expected_epoch=evidence['epoch'], actual_epoch=state.get('epoch'),
                  rt_timestamp=state.get('state', {}).get('timestamp'), age_ms=state.get('age_ms'),
                  joint_delta_rad=delta.tolist(), max_joint=('left' if index < 7 else 'right')+f' J{index%7+1}',
                  max_joint_delta_rad=float(delta[index]), limit_rad=JOINT_START_LIMIT_RAD,
                  expected_gripper=evidence['gripper_feedback'], actual_gripper=grip.tolist())
    evidence['start_guard_failure'] = detail
    message = json.dumps(detail, ensure_ascii=False, allow_nan=False)
    print('起点校验拒绝：'+message, flush=True)
    raise RuntimeError('记录起点后状态变化；'+message)


class SettledJoints:
    """Existing fresh velocity window plus a small joint-position drift window."""
    def __init__(self, window, read_arrays, epoch):
        self.window, self.read_arrays, self.epoch = window, read_arrays, epoch
        self.anchor = None
        self.grip = None
        self.latest = None

    def observe(self, state, now):
        if state.get('epoch') != self.epoch:
            raise RuntimeError('连续停稳期间控制会话变化；不继续下一段')
        joints, _, grip = self.read_arrays(state)
        drift = 0. if self.anchor is None else float(np.max(np.abs(joints-self.anchor)))
        if (self.anchor is None or self.window.start is None or drift > SETTLED_DRIFT_RAD
                or not np.array_equal(grip, self.grip)):
            self.window.reset()
            self.anchor, self.grip = joints.copy(), grip.copy()
            drift = 0.
        settled = self.window.observe(state, now)
        self.latest = dict(joints_rad=joints.tolist(), gripper=grip.tolist(),
                           drift_rad=drift, drift_limit_rad=SETTLED_DRIFT_RAD,
                           stable_samples=self.window.samples, velocity=self.window.latest,
                           rt_timestamp=state.get('state', {}).get('timestamp'), settled=bool(settled))
        return settled


def check_settled_j7(evidence, joints):
    stage = evidence.get('stages', [{}])[0]
    if stage.get('stage') not in ('rotate_j6_only', 'rotate_j7_fingers_y', 'rotate_j7_only', 'j6_axis_align'):
        return
    actual = np.asarray(joints, float)
    goal = np.asarray(stage['target_joints_rad'], float)
    delta = actual-goal
    fixed = np.arange(14)!=(13 if stage['stage'] in ('rotate_j7_fingers_y','rotate_j7_only') else 12)
    evidence['settled_j6_check'] = dict(delta_rad=delta.tolist(), joint_limit_deg=.5)
    if np.max(np.abs(delta)) > np.deg2rad(.5):
        raise RuntimeError('J6调整/旋转停稳后关节到位误差超限')
    if stage['stage'] in ('rotate_j6_only','rotate_j7_fingers_y','rotate_j7_only') and np.max(np.abs(delta[fixed])) > .002:
        raise RuntimeError('仅J6轨迹后其他关节保持误差超限')
