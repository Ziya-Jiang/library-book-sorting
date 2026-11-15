"""Measured fixed-axis guard for opt-in plug Cartesian motions (metres)."""
import numpy as np


def check_axes(state, target, axes):
    try:
        actual = np.asarray(state['right_arm']['position'], float)
        target = np.asarray(target, float)
        valid = actual.shape == (3,) and target.shape == (3,) and np.isfinite(actual).all() and np.isfinite(target).all()
        error = float(np.linalg.norm((actual-target)[list(axes)])*1000) if valid else float('inf')
    except (KeyError, TypeError, ValueError):
        error = float('inf')
    if error > 3.:
        print(f'插槽固定轴跟踪失败：axes={axes}, error={error:.3f}mm, limit=3mm', flush=True)
        return False
    return True
