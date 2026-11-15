#!/usr/bin/env python3
"""Restore only immutable bootstrap assets. Never contact or start the robot."""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'robot_cockpit/dual_arm_j6_continuous'

def install_bytes(path, data, mode=0o644):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f'Refusing to replace different existing file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.chmod(mode)
        # Exclusive link prevents replacing an existing file after the precheck.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def binary_bytes():
    data = lzma.decompress((ROOT/'deployment/seldon_rt_control.xz').read_bytes())
    meta = json.loads((ROOT/'deployment/rt_binary.json').read_text())
    if hashlib.sha256(data).hexdigest() != meta['sha256']:
        raise RuntimeError('RT binary digest mismatch')
    return data

def restore(root):
    root = Path(root).resolve()
    profile = root/PROFILE
    if not (profile/'right_approach.py').is_file():
        raise RuntimeError('Target must already contain this source checkout')
    install_bytes(profile/'runtime/build/seldon_rt_control', binary_bytes(), 0o755)
    install_bytes(profile/'runtime/build/manifest.json', (ROOT/'deployment/rt_build_manifest.json').read_bytes())
    install_bytes(profile/'runtime/active_release.json', (ROOT/'deployment/active_release.json').read_bytes())
    for name in ('dual_arm_control/runtime','launcher/runtime/book_grasp'):
        (root/'robot_cockpit'/name).mkdir(parents=True, exist_ok=True)
    print(f'Restored immutable assets under {profile}; no services started.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    restore(parser.parse_args().root)
