"""Offline logging-only RT build from the exact CAN-feedback build; no hardware execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from patch_sources import patch_rt

BASE = Path('/mnt/data/erazer/rt_can_feedback_20261104/candidate')
ROBOT = Path('/home/erazer/Seldon/dual_arm_cockpit')
BUILD = Path('/home/erazer/cuarm_panel_control/cuarm_rt_control/build')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    assets = Path(__file__).resolve().parent
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads((BASE / 'manifest.json').read_text())
    assert sha(BASE / 'seldon_rt_control') == base['candidate_sha256']
    for name, digest in base['sources'].items():
        assert sha(BASE / 'source' / name) == digest, name
    # Reproduce all retained objects/libraries before changing only the RT entry TU.
    link = base['link_command'].copy()
    link[link.index('-o') + 1] = str(output / 'base_relinked')
    subprocess.run(link, cwd=BUILD, check=True)
    assert sha(output / 'base_relinked') == base['candidate_sha256'], 'Installed build lineage changed'
    src = output / 'source'
    src.mkdir(exist_ok=True)
    (src / 'rt_control.cpp').write_text(patch_rt((BASE / 'source/rt_control.cpp').read_text()))
    shutil.copy2(assets / 'rt_log_file.h', src / 'rt_log_file.h')
    flags = (BUILD / 'CMakeFiles/dual_v2_2_rt_control.dir/flags.make').read_text()
    values = dict(line.split(' = ', 1) for line in flags.splitlines() if ' = ' in line)
    command = ['c++', '-I' + str(src), '-I' + str(BASE / 'source')]
    for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
        command += shlex.split(values[key])
    hand = ROBOT / 'config_sources/V2/dual_v2_2_control/cuarm_rt_control/hardware_modules/eyou-main/include'
    command += ['-I' + str(ROBOT / 'rt'), '-I' + str(hand), '-I' + str(hand / 'x86'),
                '-c', str(src / 'rt_control.cpp'), '-o', str(output / 'rt_control.o')]
    subprocess.run(command, cwd=BUILD, check=True)
    link = [str(output / 'rt_control.o') if item == str(BASE / 'rt_control.o') else item
            for item in base['link_command']]
    link[link.index('-o') + 1] = str(output / 'seldon_rt_control')
    subprocess.run(link, cwd=BUILD, check=True)
    manifest = dict(base_binary_sha256=base['candidate_sha256'], base_reproduced=True,
                    candidate_sha256=sha(output / 'seldon_rt_control'), link_command=link,
                    sources={p.name: sha(p) for p in src.iterdir()},
                    source_base=str(BASE), scope='logging only; original driver libraries and control code retained')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('link_command', 'sources')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'candidate')
    build(parser.parse_args().output)
