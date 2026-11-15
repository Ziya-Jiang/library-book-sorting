"""URDF model manifest. Only allow explicitly referenced local mesh files."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

JOINTS = [[f'arm_J{j}_{side}' for j in range(1, 8)] for side in ('l', 'r')]

def vector(node, attr, default):
    return [float(x) for x in node.get(attr, default).split()] if node is not None else [float(x) for x in default.split()]

def origin(node):
    return {'xyz': vector(node, 'xyz', '0 0 0'), 'rpy': vector(node, 'rpy', '0 0 0')}

class Model:
    def __init__(self, urdf, mesh_root, verified=False):
        self.urdf, self.root = Path(urdf).resolve(), Path(mesh_root).resolve()
        doc = ET.parse(self.urdf).getroot()
        self.files, links, joints = {}, [], []
        names = {j.get('name') for j in doc.findall('joint')}
        if not set(sum(JOINTS, [])).issubset(names):
            raise ValueError('URDF must contain all dual_v2_2 arm_J1..7_l/r joints')
        for link in doc.findall('link'):
            visuals = []
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    continue
                filename = mesh.get('filename')
                relative = filename.split('dual_v2_2/', 1)[-1] if 'dual_v2_2/' in filename else filename
                candidates = [self.root / relative, self.urdf.parent / relative]
                # The supplied dual_v2_2 full-body URDF predates the mesh/base move.
                # Resolve this specific layout change within the selected model only.
                if Path(relative).parent == Path('meshes'):
                    candidates.append(self.root / 'meshes/base' / Path(relative).name)
                path = next((p.resolve() for p in candidates if p.is_file()), None)
                if path is None or not (path.is_relative_to(self.root) or path.is_relative_to(self.urdf.parent)):
                    raise ValueError(f'Missing model mesh: {filename}')
                key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
                self.files[key] = path
                material = visual.find('material/color')
                visuals.append(dict(mesh='/mesh/'+key, **origin(visual.find('origin')),
                    scale=vector(mesh, 'scale', '1 1 1'), color=vector(material, 'rgba', '0.72 0.78 0.86 1')))
            links.append({'name': link.get('name'), 'visuals': visuals})
        for joint in doc.findall('joint'):
            limit = joint.find('limit')
            joints.append(dict(name=joint.get('name'), type=joint.get('type'),
                parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
                axis=vector(joint.find('axis'), 'xyz', '1 0 0'), **origin(joint.find('origin')),
                limits=[float(limit.get('lower', '-6.283185')), float(limit.get('upper', '6.283185'))] if limit is not None else None))
        untracked = [j['name'] for j in joints if j['type'] != 'fixed' and j['name'] not in sum(JOINTS, [])]
        self.manifest = dict(links=links, joints=joints, arm_joints=JOINTS, source=str(self.urdf),
            robot_name=doc.get('name'), untracked_joints=untracked, mesh_root=str(self.root),
            sha256=hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
            source_note=('操作端同步 · '+doc.get('name', '') + (' · 非双臂关节无反馈，显示参考位' if untracked else '')) if verified else '机器人现有 dual_v2_2；尚未与操作端最新模型核对')
