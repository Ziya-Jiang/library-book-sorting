"""Export the configured planner's CPU model/code without opening robot sockets."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path('/mnt/ssd/user/V/dual_v2_2_control/cuarm_upper_planner')
PROFILE=Path(__file__).resolve().parents[1]/'backend'


def export(planner,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    cpu=planner._v31_bridge.cpu;collision=planner._v31_direct.collision
    def convert(g):
        p=planner
        return np.asarray(p.gripper_convertion(list(g),p.robot_sent_gripper_type,p.robot_sent_gripper_urdf_order,
            p.robot_sent_gripper_normalized,p.robot_sent_gripper_zero_open,p.curobo_used_gripper_type,
            p.curobo_used_gripper_urdf_order,p.curobo_used_gripper_normalized,p.curobo_used_gripper_zero_open),float)
    bias=convert(np.zeros(2));matrix=np.stack([convert(np.eye(2)[i])-bias for i in range(2)],axis=1)
    for t in (.13,.5,.91):
        g=np.linspace(0,t,2)
        if not np.allclose(convert(g),matrix@g+bias,atol=1e-8):raise ValueError('Non-affine gripper conversion cannot export')
    data=dict(deployment_hashes={name:hashlib.sha256((PROFILE/name).read_bytes()).hexdigest()
                                for name in ('planner_bridge_board.py','server_board.py','v31_timing.py','v31_wire.py',
                                             'planner_bridge_visual.py','controller_v31.py','rt_logs.py','stop_diagnostics.py','server_v31.py',
                                             'packet_trace.py','active_arm_policy.py','gripper_policy.py','planner_bridge_v31.py',
                                             'heartbeat_policy.py','startup_reservation.py')},
        contract={k:v['sha256'] for k,v in json.loads((PROFILE/'contract.json').read_text()).items()},
        fk={k:getattr(cpu,k) for k in ('fixed','parent','joint','kind','offset','store','dof')},
        collision={k:getattr(collision,k) for k in ('spheres','links','inverse','offset','pairs','bounds','boxes')},
        arm_order=planner._v31_bridge.arm_order,
        full_order=planner.reorder_input(list(range(14)),list(range(14,16)),as_curobo=True),
        arm_limits=planner.arm_soft_joint_limits,target_links=planner.target_link_names,
        tcp_offsets=planner.end_effector_offset,settings=planner._v31_direct.c,
        grip_matrix=matrix,grip_bias=bias)
    def default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (root/'model.json').write_text(json.dumps(data,default=default,sort_keys=True,allow_nan=False))
    for path in (ROOT/'seldon_v31/board_servo').glob('*.py'):shutil.copy2(path,root/path.name)
    for src,dst in [('v3_1_cpu_fk.py','cpu_fk.py'),('v3_1_direct_servo.py','kernel.py'),('v3_1_execution.py','execution.py')]:
        shutil.copy2(ROOT/'scripts'/src,root/dst)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix in ('.py','.json') and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def main():
    import argparse,sys
    sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    import dual_v2_2_upper_planner_v3_1 as old
    config=old.CuroboConfig.load_from_yaml_path(str(old.CONFIG_PATH))
    p=old.create_curobo_planner(config,old.PlannerCuroboSolverType.MOTION_GEN)
    try:print('BOARD_BUNDLE_SHA='+export(p,args.directory))
    finally:p.terminate()

if __name__=='__main__':main()
