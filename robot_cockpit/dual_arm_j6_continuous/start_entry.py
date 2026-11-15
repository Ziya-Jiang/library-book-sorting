"""Switch only an idle local planner; reuse unchanged installed dual-arm RT."""
import importlib.util
from pathlib import Path
import sys
import start


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print('right_arm_j6_continuous_start.sh [--restart-planner | --stop-planner | --check]\n'
              '准备独立J6 Planner及现有双臂服务。空闲时正常退出原双臂Planner；不接管运动。\n'
              '--stop-planner 只停止本入口的空闲Planner，之后可启动原dual_arm_start.sh。')
        return
    if '--stop-planner' in sys.argv:
        if len(sys.argv)!=2:raise ValueError('--stop-planner不可与其他参数组合')
        start.ensure_tunnel()
        start.restart_idle_planner(start.existing_planner())
        print('J6 Planner已停止；RT、相机与网页保留。')
        return
    if any(x not in ('--restart-planner','--check') for x in sys.argv[1:]):
        raise ValueError('只支持 --restart-planner、--stop-planner、--check')
    if '--check' not in sys.argv:
        start.ensure_tunnel()
        for profile in ('dual_arm_control','dual_arm_j6_trial'):
            source=Path(__file__).parent.parent/profile/'start.py'
            spec=importlib.util.spec_from_file_location('previous_'+profile,source)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            pid=old.existing_planner()
            if pid is not None:
                if '--restart-planner' not in sys.argv:
                    raise RuntimeError('旧Planner仍运行；请结束控制后用 --restart-planner 切换连续版本')
                old.restart_idle_planner(pid)
    start.main()

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as e:
        print('连续轨迹启动未完成：'+str(e),file=sys.stderr);raise SystemExit(1)
