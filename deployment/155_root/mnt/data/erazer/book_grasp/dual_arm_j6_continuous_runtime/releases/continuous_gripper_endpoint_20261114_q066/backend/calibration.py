"""Coordinate bridge between the running legacy RT and supplied V2 calibration.
Legacy decoder: q_rt = sign_rt * (encoder - init_rt).
V2 decoder:     q_v2 = sign_v2 * encoder - init_v2.
Both formulas are verified against their RobotArms::joints_to_mdh source.
"""
import copy
import hashlib
import math
from pathlib import Path
import yaml

class JointCalibration:
    def __init__(self, gain=None, offset=None, provenance=None):
        self.gain = gain if gain is not None else [[1]*7 for _ in range(2)]
        self.offset = offset if offset is not None else [[0.0]*7 for _ in range(2)]
        if len(self.gain)!=2 or len(self.offset)!=2:
            raise ValueError('Calibration must contain two arms')
        for gains, offsets in zip(self.gain,self.offset):
            if len(gains)!=7 or len(offsets)!=7 or any(g not in (-1,1) for g in gains) or any(not math.isfinite(b) for b in offsets):
                raise ValueError('Invalid joint calibration')
        self.provenance=provenance or {}

    @classmethod
    def from_configs(cls, legacy, current):
        old=yaml.safe_load(Path(legacy).read_text())['kinematics']
        new=yaml.safe_load(Path(current).read_text())['kinematics']
        gain,offset=[],[]
        for side in ('left_arm','right_arm'):
            so,sn=old[side]['sign'],new[side]['sign']
            zo,zn=old[side]['init_mdhpos'],new[side]['init_mdhpos']
            if any(len(v)!=7 for v in (so,sn,zo,zn)):
                raise ValueError('Expected seven encoder calibration values')
            gain.append([b*a for a,b in zip(so,sn)])
            offset.append([b*x-y for b,x,y in zip(sn,zo,zn)])
        provenance={label:{'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for label,path in [('legacy',legacy),('v2',current)]}
        return cls(gain,offset,provenance)

    def to_v2(self,a,j,q):return self.gain[a][j]*q+self.offset[a][j]
    def to_rt(self,a,j,q):return (q-self.offset[a][j])/self.gain[a][j]
    def positions(self,rows):return [[self.to_v2(a,j,q) for j,q in enumerate(row)] for a,row in enumerate(rows)]
    def bounds(self,a,j,pair,position=True):
        return sorted(self.gain[a][j]*q+(self.offset[a][j] if position else 0) for q in pair)
    def manifest(self):return dict(gain=self.gain,offset_rad=self.offset,provenance=self.provenance)
    def state_for_web(self,state):
        result=copy.deepcopy(state)
        result['coordinate_system']='V2 calibrated'
        for a,arm in enumerate(result['arms']):
            arm['rt_position']=arm['position'][:]
            for key in ('position','target'):
                arm[key]=[self.to_v2(a,j,q) for j,q in enumerate(arm[key])]
            for key in ('velocity','torque'):
                arm[key]=[self.gain[a][j]*q for j,q in enumerate(arm[key])]
            for key in ('soft_position','hard_position','soft_velocity','hard_velocity','soft_torque','hard_torque'):
                arm[key]=[self.bounds(a,j,pair,key.endswith('position')) for j,pair in enumerate(arm[key])]
            # Legacy FK uses a different URDF; don't label it as calibrated V2 FK.
            for key in ('joint_pose','ee_pose','tool_pose'):
                arm['rt_'+key]=arm.pop(key)
        return result
