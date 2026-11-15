"""Two identical tools grafted onto the installed, calibrated RT arm assembly."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import yaml


def prepare_runtime_config(root, base, cfg, panel_port, rt_port, simulation):
    root,base,cfg=map(Path,(root,base,cfg));cfg.mkdir(parents=True,exist_ok=True)
    profile=root/'configuration'
    robot=ET.parse(base/'share/no_gripper/dual_arm.urdf').getroot()
    tools=ET.parse(profile/'dual_gripper.urdf').getroot()
    for tag in ('link','joint'):
        for elem in tools.findall(tag):
            if elem.get('name','').startswith('gripper_'):robot.append(copy.deepcopy(elem))
    # Resolve each source's own mesh root, without changing any arm kinematics.
    for mesh in robot.findall('.//mesh'):
        name=mesh.get('filename')
        if name.startswith('meshes/gripper/'):
            mesh.set('filename',str(profile/name))
    robot.set('name','dual_gripper_rt')
    ET.ElementTree(robot).write(cfg/'dual_gripper.urdf',encoding='utf-8',xml_declaration=True)
    assembly=yaml.safe_load((base/'share/config/assembly/no_gripper.yaml').read_text())
    assembly.update(name='dual_gripper_rt',urdf='dual_gripper.urdf',
                    import_yaml=str(base/'share/config/components/dual_v2_2.yaml'))
    assembly['gripper']=yaml.safe_load((profile/'grippers.yaml').read_text())
    if [g['joint_size'] for g in assembly['gripper']] != [1,1]:raise ValueError('Wrong dual-gripper profile')
    for g in assembly['gripper']:
        for key in ('joint_soft_limit','joint_hard_limit'):
            g['safety'][key]['position']=[v[0] for v in g['safety'][key]['position']]
    (cfg/'robot.yaml').write_text(yaml.safe_dump(assembly,sort_keys=False))
    config=yaml.safe_load((base/'config.yaml').read_text())
    config['robot']={'import_yaml':'robot.yaml'};config['curobo']={}
    config['panel'].update(ip='127.0.0.1',port=panel_port)
    config['rt_control'].update(ip='127.0.0.1',port=rt_port,hardware_simulation=simulation)
    (cfg/'config.yaml').write_text(yaml.safe_dump(config))
    return config
