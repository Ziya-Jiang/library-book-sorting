"""Fixed independent-trial reset; never accumulate a per-call Cartesian offset."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from backend.board_servo.model import Model

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'task_config/right_first_reset.json'


def load_target(path=CONFIG):
    data=json.loads(Path(path).read_text())
    model=Model(ROOT/'backend/board_servo')
    if data.get('schema')!='right-trial-reset-v1' or data.get('model_digest')!=model.digest:
        raise ValueError('独立first_reset配置与模型不匹配；未发送复位')
    angles=np.asarray(data['right_joint_deg'],float)
    if angles.shape!=(7,) or not np.isfinite(angles).all():
        raise ValueError('独立first_reset关节目标无效')
    q=np.deg2rad(angles)
    if np.any(q<model.bounds[0,7:]) or np.any(q>model.bounds[1,7:]):
        raise ValueError('独立first_reset关节目标超限')
    position,rotation=model.tcp(np.r_[np.zeros(7),q])
    target=np.asarray(data['target_tcp_mm'],float)
    expected=np.asarray(data['target_wxyz'],float)
    if target.shape!=(3,) or expected.shape!=(4,) or not np.isfinite(np.r_[target,expected]).all():
        raise ValueError('独立first_reset末端参考无效')
    error=np.linalg.norm(position[0,1]*1000-target)
    angular=(Rotation.from_quat(rotation[0,1,[1,2,3,0]]).inv()*Rotation.from_quat(expected[[1,2,3,0]])).magnitude()
    if error>.1 or angular>np.deg2rad(.05):
        raise ValueError('独立first_reset关节与末端配置不一致')
    return angles.copy(),data
