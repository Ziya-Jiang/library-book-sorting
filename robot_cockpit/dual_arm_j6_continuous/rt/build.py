"""Build independent RT from pinned working driver objects; never start hardware."""
from pathlib import Path
import hashlib,json,subprocess,shlex
ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/data/erazer/book_grasp/diagnostics/rt_log_rotation_20261105/candidate')
BUILD=Path('/home/erazer/cuarm_panel_control/cuarm_rt_control/build')
COCKPIT=Path('/home/erazer/Seldon/dual_arm_cockpit')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    output=ROOT.parent/'runtime/build';output.mkdir(parents=True,exist_ok=True)
    base=json.loads((BASE/'manifest.json').read_text())
    if sha(BASE/'seldon_rt_control')!=base['candidate_sha256']:raise ValueError('RT base differs')
    # Reproduce the installed binary before using retained link objects.
    link=list(base['link_command']);link[link.index('-o')+1]=str(output/'base_relinked')
    subprocess.run(link,cwd=BUILD,check=True)
    if sha(output/'base_relinked')!=base['candidate_sha256']:raise ValueError('Driver objects changed')
    (output/'base_relinked').unlink()
    flags=(BUILD/'CMakeFiles/dual_v2_2_rt_control.dir/flags.make').read_text()
    values=dict(l.split(' = ',1) for l in flags.splitlines() if ' = ' in l)
    command=['c++','-I'+str(ROOT),'-I'+str(COCKPIT/'rt')]
    for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):command+=shlex.split(values[key])
    hand=COCKPIT/'config_sources/V2/dual_v2_2_control/cuarm_rt_control/hardware_modules/eyou-main/include'
    command+=['-I'+str(hand),'-I'+str(hand/'x86'),'-c',str(ROOT/'rt_control.cpp'),'-o',str(output/'rt_control.o')]
    subprocess.run(command,cwd=BUILD,check=True)
    link=[str(output/'rt_control.o') if v==str(BASE/'rt_control.o') else v for v in base['link_command']]
    link[link.index('-o')+1]=str(output/'seldon_rt_control')
    subprocess.run(link,cwd=BUILD,check=True)
    m={'profile':'dual_gripper','base_sha256':base['candidate_sha256'],'base_reproduced':True,
       'binary_sha256':sha(output/'seldon_rt_control'),'sources':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},
       'link_command':link,'hardware_started':False}
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({k:v for k,v in m.items() if k not in ('sources','link_command')}))
if __name__=='__main__':main()
