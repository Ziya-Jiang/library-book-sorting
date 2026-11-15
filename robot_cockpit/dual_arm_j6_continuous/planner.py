"""CuRobo for the independent dual-gripper profile; old planners untouched."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLANNER=ROOT.parents[1]/'dual_v2_2_control/cuarm_upper_planner'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path)
    p.add_argument('--profile-root',type=Path,help='Export only: isolated candidate configuration/backend root')
    p.add_argument('--check-config',action='store_true')
    a=p.parse_args()
    if a.profile_root and not a.export:p.error('--profile-root requires --export')
    profile=(a.profile_root or ROOT).resolve()
    os.environ['CUARM_CONFIG_ROOT']=str(profile/'configuration')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    sys.path.insert(0,str(PLANNER))
    import dual_v2_2_upper_planner_v3_1 as base
    # Prefer only our isolated protocol/export adapters; retain ordinary solver,
    # transport deadlines, limits and collision checks from the existing planner.
    sys.path[:0]=[str(ROOT/'planner_scripts'),str(ROOT/'backend')]
    import v3_1_seldon as remote
    import v3_1_board_export as exporter
    from planning_diagnostics import install as install_diagnostics
    install_diagnostics(base)
    base.ARGUMENTS=argparse.Namespace(check_config=a.check_config,robot_host='10.30.3.155',planner_bind='10.30.3.100')
    import v3_1_readiness as ready
    ready.receipt_path=lambda port: Path(f'/tmp/cuarm-dual-arm-j6-continuous-{os.getuid()}-{int(port)}.json')
    remote.install_remote_path()
    if a.export:
        exporter.PROFILE=profile/'backend'
        c=base.CuroboConfig.load_from_yaml_path(str(base.CONFIG_PATH))
        planner=base.create_curobo_planner(c,base.PlannerCuroboSolverType.MOTION_GEN)
        try:print('DUAL_GRIPPER_BUNDLE_SHA='+exporter.export(planner,a.export))
        finally:planner.terminate()
        return
    from v3_1_board_client import install
    install()
    return base.main()
if __name__=='__main__':raise SystemExit(main())
