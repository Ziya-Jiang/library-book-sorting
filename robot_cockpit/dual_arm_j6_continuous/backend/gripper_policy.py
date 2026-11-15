"""Validate the installed two-single-axis grippers; never infer an absent left tool."""
import math

def validate_gripper_targets(controller, command):
    targets=command.get('grippers')
    feedback=controller.state.get('grippers')
    if not isinstance(targets,list) or len(targets)!=2 or not isinstance(feedback,list) or len(feedback)!=2:
        raise ValueError('Installed dual-gripper profile requires two grippers')
    for index,(values,state) in enumerate(zip(targets,feedback)):
        if not isinstance(values,list) or len(values)!=1:
            raise ValueError('Each installed gripper requires exactly one target')
        value=values[0]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Gripper target must be a finite number')
        bounds=[]
        for field in ('soft_position','hard_position'):
            rows=state.get(field)
            if not isinstance(rows,list) or len(rows)!=1 or len(rows[0])!=2:
                raise ValueError('Gripper feedback limit shape invalid')
            lo,hi=rows[0]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lo,hi)) or lo>hi:
                raise ValueError('Gripper feedback limits invalid')
            bounds.append((lo,hi))
        lower=max(0.,*(b[0] for b in bounds));upper=min(1.,*(b[1] for b in bounds))
        if not lower<=value<=upper:
            raise ValueError(f'V3.1 gripper {index} outside limits')
