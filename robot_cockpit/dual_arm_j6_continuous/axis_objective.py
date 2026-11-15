"""Differentiable base-Y J6 axis cost for a dedicated CuRobo GPU IK solver.

Uses the hash-checked exported CuRobo chain, not a second URDF or calibration.
Only preparation IK installs this cost; ordinary motion solvers are untouched.
"""
import copy
from types import MethodType
import numpy as np
import torch

AXIS_COST_WEIGHT = 50000.0


@torch.jit.script
def chain_axis(q: torch.Tensor, fixed: torch.Tensor, indices: torch.Tensor,
               vectors: torch.Tensor, gains: torch.Tensor, biases: torch.Tensor,
               final_axis: torch.Tensor):
    # q: arbitrary batch/horizon followed by DOF. Only rotations affect an axis.
    flat = q.reshape(-1, q.shape[-1])
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    r = eye.unsqueeze(0).expand(flat.shape[0], 3, 3)
    for i in range(indices.shape[0]):
        r = torch.matmul(r, fixed[i])
        a = flat[:, indices[i]] * gains[i] + biases[i]
        v = vectors[i]
        x, y, z = v[0], v[1], v[2]
        zero = x * 0.0
        skew = torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
        outer = v.unsqueeze(1) * v.unsqueeze(0)
        rot = eye + torch.sin(a)[:,None,None]*skew + (1.0-torch.cos(a))[:,None,None]*(outer-eye)
        r = torch.matmul(r, rot)
    r = torch.matmul(r, fixed[-1])
    return torch.matmul(r, final_axis).reshape(q.shape[:-1] + [3])


class AxisObjective:
    def __init__(self, model, names, arm_names, tensor_args):
        cpu = model.cpu
        cpu_joint = list(model.arm_order).index(12)
        hits = np.flatnonzero((cpu.joint == cpu_joint) & (cpu.kind >= 3))
        if len(hits) != 1:
            raise ValueError('Cannot uniquely bind right J6 axis to exported model')
        end = int(hits[0]); chain=[]; i=end
        while i:
            chain.append(i);i=int(cpu.parent[i])
        chain.append(0);chain.reverse()
        # Fold fixed links into the next moving joint. The final transform ends
        # BEFORE J6 rotates, so neither J6 nor downstream J7 changes this axis.
        fixed=[];indices=[];vectors=[];gains=[];biases=[];r=np.eye(3)
        for i in chain:
            r=r@cpu.fixed[i,:3,:3]
            if i == end:
                fixed.append(r);break
            kind=int(cpu.kind[i])
            if kind == -1:continue
            if kind not in (3,4,5):
                raise ValueError('Unexpected prismatic joint before J6')
            fixed.append(r);r=np.eye(3)
            arm_index=int(model.arm_order[int(cpu.joint[i])])
            indices.append(names.index(arm_names[arm_index]))
            vectors.append(np.eye(3)[kind-3])
            gains.append(cpu.offset[i,0]);biases.append(cpu.offset[i,1])
        self.fixed=tensor_args.to_device(np.asarray(fixed))
        self.indices=torch.tensor(indices,device=tensor_args.device,dtype=torch.long)
        self.vectors=tensor_args.to_device(np.asarray(vectors))
        self.gains=tensor_args.to_device(np.asarray(gains))
        self.biases=tensor_args.to_device(np.asarray(biases))
        self.final_axis=tensor_args.to_device(np.eye(3)[int(cpu.kind[end])-3]*np.sign(cpu.offset[end,0]))

    def axis(self,q):
        return chain_axis(q,self.fixed,self.indices,self.vectors,self.gains,self.biases,self.final_axis)

    def cost(self,q):
        axis=self.axis(q)
        # sin(angle)^2: both +Y and -Y accepted; differentiable, no acos near 0.
        return AXIS_COST_WEIGHT*(axis[...,0].square()+axis[...,2].square())


def build_axis_solver(planner, model):
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    source=planner._motion_gen.ik_solver
    solver=IKSolver(IKSolverConfig.load_from_robot_config(
        copy.deepcopy(source.robot_config), tensor_args=planner.tensor_args,
        world_coll_checker=source.world_coll_checker, num_seeds=source.num_seeds,
        position_threshold=source.position_threshold, rotation_threshold=source.rotation_threshold,
        grad_iters=200, use_particle_opt=True, use_cuda_graph=True,
        self_collision_check=True, self_collision_opt=True,collision_activation_distance=0.))
    seen=set()
    for rollout in solver.get_all_rollout_instances():
        if id(rollout) in seen:continue
        seen.add(id(rollout))
        objective=AxisObjective(model,list(rollout.joint_names),list(planner.total_arm_joint_names),planner.tensor_args)
        original=rollout.cost_fn
        def cost_fn(self,state,action_batch=None,_original=original,_objective=objective):
            base=_original(state,action_batch)
            extra=_objective.cost(state.state_seq.position)
            if self.sum_horizon:extra=extra.sum(dim=-1)
            return base+extra
        rollout.cost_fn=MethodType(cost_fn,rollout)
        rollout._j6_axis_objective=objective
    return solver


def warm_axis_solver(solver):
    """Compile/capture before readiness; this never publishes a trajectory."""
    from curobo.types.math import Pose
    q=solver.get_retract_config().reshape(1,-1)
    state=solver.fk(q)
    solver.solve_single(Pose(state.ee_position,state.ee_quaternion),
                        retract_config=q,seed_config=q[:,None,:],
                        return_seeds=solver.num_seeds,link_poses=state.link_poses)
    torch.cuda.synchronize(solver.tensor_args.device)
