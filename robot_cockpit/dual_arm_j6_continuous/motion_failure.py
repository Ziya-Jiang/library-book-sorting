"""Retain execution evidence without converting it to a planning rejection."""
import json
import time
import numpy as np

class MotionFailure(RuntimeError):
    def __init__(self, failure):
        self.failure = dict(failure) if isinstance(failure, dict) else {'reason': 'missing_failure_evidence'}
        stopped_at_minimum = self.failure.get('retry_stopped') == 'minimum_segment_length'
        phase = ('执行反馈失败' if self.failure.get('execution_started') is True
                 else '规划拒绝' if stopped_at_minimum else '运动未确认')
        suffix = ('；已到细分下限，停止后续动作' if stopped_at_minimum
                  else '；停止后续动作，不自动重试')
        super().__init__(phase+'：'+json.dumps(self.failure, ensure_ascii=False, default=str)+suffix)



def check_orientation(robot, state, reference, tolerance_rad, stage):
    actual, reference = np.asarray(state.get('right_arm', {}).get('orientation', []), float), np.asarray(reference, float)
    valid = (actual.shape == reference.shape == (4,) and np.isfinite([actual, reference]).all()
             and min(np.linalg.norm(actual), np.linalg.norm(reference)) > 1e-8)
    angle = (2*np.arccos(np.clip(abs(actual@reference)/(np.linalg.norm(actual)*np.linalg.norm(reference)),0,1))
             if valid else float('inf'))
    if angle <= tolerance_rad:
        return True
    robot.last_motion_failure = dict(reason='lower_orientation_drift', stage=stage,
        failure_kind='measured_orientation_drift', execution_started=True, unix_s=time.time(),
        orientation_error_deg=float(np.rad2deg(angle)) if valid else None,
        orientation_limit_deg=float(np.rad2deg(tolerance_rad)),
        measured_wxyz=actual.tolist(), target_wxyz=reference.tolist(),
        hold_requested=False, hold_confirmed=False)
    robot.target_command.description = 'book_v3_1_hold'
    try:
        robot.udp.send(robot.target_command)
        robot.last_motion_failure['hold_requested'] = True
    except Exception as error:
        robot.last_motion_failure['hold_send_error'] = f'{type(error).__name__}: {error}'
        raise MotionFailure(robot.last_motion_failure) from error
    print('运动姿态保持失败：'+json.dumps(robot.last_motion_failure,ensure_ascii=False),flush=True)
    print('已请求保持，尚未确认停稳；取消后续动作。',flush=True)
    return False
