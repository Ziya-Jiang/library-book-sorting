"""Portable, hash-checked CuRobo CPU geometry export; no pickle or CUDA load."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .cpu_fk import CpuKinematics
from .kernel import CpuCollision


def verify_bundle(directory):
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        if Path(name).name!=name or name=='manifest.json':raise ValueError('Invalid bundle path')
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Board servo bundle changed: '+name)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


class Model:
    def __init__(self,directory):
        self.digest=verify_bundle(directory)
        data=json.loads((Path(directory)/'model.json').read_text())
        self.contract=data['contract'];self.settings=data['settings']
        self.deployment_hashes=data['deployment_hashes']
        cpu=CpuKinematics.__new__(CpuKinematics)
        for key in ('fixed','parent','joint','kind','offset'):
            setattr(cpu,key,np.asarray(data['fk'][key]))
        cpu.store=data['fk']['store'];cpu.dof=data['fk']['dof'];self.cpu=cpu
        self.arm_order=np.asarray(data['arm_order'],int)
        self.full_order=np.asarray(data['full_order'],int)
        self.links=data['target_links'];self.offsets=np.asarray(data['tcp_offsets'])
        self.bounds=np.asarray(data['arm_limits'])
        self.grip_matrix=np.asarray(data['grip_matrix']);self.grip_bias=np.asarray(data['grip_bias'])
        collision=CpuCollision.__new__(CpuCollision);collision.cpu=cpu
        for key in ('spheres','links','inverse','offset','pairs','bounds'):
            value=np.asarray(data['collision'][key])
            if key in ('links','inverse','pairs'):value=value.astype(int)
            setattr(collision,key,value)
        collision.boxes=[tuple(np.asarray(x) for x in box) for box in data['collision']['boxes']]
        self.collision=collision

    def tcp(self,joints):
        q=np.atleast_2d(joints)[:,self.arm_order]
        p,r=self.cpu.poses(q,self.links)
        for i,offset in enumerate(self.offsets):
            p[:,i]+=Rotation.from_quat(r[:,i,[1,2,3,0]]).apply(offset)
        return p,r

    def ordered(self,joints,grippers):
        return np.r_[joints,grippers][self.full_order]
