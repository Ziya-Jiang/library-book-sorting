"""Hardware identity checks for the installed-both dual-gripper profile."""
from pathlib import Path
import json,hashlib

def verify(root):
    root=Path(root)
    m=json.loads((root/'profile_manifest.json').read_text())
    if m.get('profile')!='dual_arm_control' or m.get('gripper_installed')!=[True,True]:
        raise ValueError('Unexpected physical tool profile')
    for name,digest in m['files'].items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()):raise ValueError('External profile code')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Profile changed: '+name)
    return m
