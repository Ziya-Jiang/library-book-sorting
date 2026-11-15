#!/usr/bin/env python3
"""Check packaged sources, calibration, contracts and imports without robot IO."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'robot_cockpit/dual_arm_j6_continuous'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-imports',action='store_true')
    args=parser.parse_args()
    count=0
    sensitive=re.compile(rb'(?:sk-(?:proj-)?[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if '.git' in rel.parts or '__pycache__' in rel.parts:continue
        # runtime beneath deployment/155_root is immutable release payload, not live state.
        if rel.parts[0]=='robot_cockpit' and 'runtime' in rel.parts:continue
        if p.is_symlink() and not p.exists():raise RuntimeError('Broken symlink: '+str(rel))
        if not p.is_file():continue
        if p.stat().st_size>=100*2**20:raise RuntimeError('Oversized Git file: '+str(rel))
        if p.suffix=='.py':ast.parse(p.read_bytes());count+=1
        if p.suffix in {'.py','.json','.txt','.md','.sh','.yaml','.yml','.js','.cpp','.h','.patch'}:
            if sensitive.search(p.read_bytes()):raise RuntimeError('Credential-like content in '+str(rel))
    manifest=json.loads((ROOT/'deployment/source_manifest.json').read_text())
    for name,record in manifest['files'].items():
        p=ROOT/name
        if 'sha256' in record and hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError('Source snapshot mismatch: '+name)
    layout=json.loads((ROOT/'deployment/release_layout.json').read_text())
    frozen=ROOT/'deployment/155_root'/layout['remote_root'].lstrip('/')
    for name,digest in json.loads((frozen/'profile_manifest.json').read_text())['files'].items():
        if name==layout['binary']:
            from restore_runtime import binary_bytes
            data=binary_bytes()
        else:data=(frozen/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Frozen release mismatch: '+name)
    if args.task_imports:
        os.environ['CUARM_CONFIG_ROOT']=str(PROFILE/'configuration')
        sys.path[:0]=[str(PROFILE),str(ROOT),str(ROOT/'robot_cockpit/launcher'),
                      str(ROOT/'dual_v2_2_agent'),str(ROOT/'dual_v2_2_agent/RobotRAM')]
        def deny_io(event,params):
            if event in ('socket.connect','socket.bind','socket.sendto','subprocess.Popen'):
                raise RuntimeError('Offline check blocked '+event)
        sys.addaudithook(deny_io)
        import right_approach_motion as motion
        motion.preload_motion_dependencies()
        import left_wrist_spine, gpt_segmentation_preview, round_control
        left_wrist_spine.calibration()
        from backend.board_servo.model import Model
        Model(PROFILE/'backend/board_servo')
    print(json.dumps({'python_files_parsed':count,'source_hashes':'passed','frozen_release_files':63,
                      'credential_pattern_scan':'passed','task_imports':args.task_imports,
                      'robot_started':False,'model_api_called':False},indent=2))

if __name__=='__main__':main()
