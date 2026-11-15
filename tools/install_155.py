#!/usr/bin/env python3
"""Install the allowlisted 155 filesystem payload; default is a read-only plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from restore_runtime import ROOT, install_bytes, binary_bytes

def payload():
    base = ROOT/'deployment/155_root'
    for path in sorted(base.rglob('*')):
        if path.is_file():
            yield path.relative_to(base), path.read_bytes(), path.stat().st_mode & 0o777
    sdk = ROOT/'deployment/camera_sdk/pyorbbecsdk'
    prefix = Path('home/erazer/Seldon/keyboard_cockpit/.venv/lib/python3.10/site-packages/pyorbbecsdk')
    for path in sorted(sdk.rglob('*')):
        if path.is_file():
            yield prefix/path.relative_to(sdk), path.read_bytes(), path.stat().st_mode & 0o777
    layout = json.loads((ROOT/'deployment/release_layout.json').read_text())
    yield Path(layout['remote_root'].lstrip('/'))/layout['binary'], binary_bytes(), 0o755

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'), help='Alternate root for an offline staging test')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--system-libs-only', action='store_true')
    parser.add_argument('--skip-system-libs', action='store_true')
    args = parser.parse_args()
    if args.system_libs_only and args.skip_system_libs:
        parser.error('Choose one library mode')
    root = args.root.resolve()
    items = [(rel,data,mode) for rel,data,mode in payload()
             if not (args.skip_system_libs and rel.parts[0]=='usr')
             and not (args.system_libs_only and rel.parts[0]!='usr')]
    conflicts = []
    for rel,data,_ in items:
        target=root/rel
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or target.read_bytes()!=data:
                conflicts.append(str(target))
    if conflicts:
        print(json.dumps({'conflicts':conflicts,'policy':'No existing different file is overwritten'},indent=2))
        return 2
    print(f'{len(items)} files, {sum(len(data) for _,data,_ in items)/2**20:.1f} MiB, target root {root}')
    if not args.apply:
        print('Plan only. Use --apply on the intended 155 host; no services are started.')
        return 0
    for rel,data,mode in items:
        install_bytes(root/rel,data,mode)
    if not args.system_libs_only:
        layout=json.loads((ROOT/'deployment/release_layout.json').read_text())
        release=root/layout['remote_root'].lstrip('/')
        for name,destination in layout['symlinks'].items():
            link=release/name
            target=root/destination.lstrip('/')
            relative=os.path.relpath(target,link.parent)
            if link.exists() or link.is_symlink():
                if link.resolve()!=target.resolve():raise RuntimeError(f'Conflicting symlink: {link}')
            else:link.symlink_to(relative,target_is_directory=True)
        manifest=json.loads((release/'profile_manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if hashlib.sha256((release/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Release checksum mismatch: {name}')
        print('All 63 release contract files verified.')
    print('Install complete; RT, Planner, cameras, grippers and arm motion were NOT started.')
    return 0

if __name__=='__main__':
    sys.exit(main())
