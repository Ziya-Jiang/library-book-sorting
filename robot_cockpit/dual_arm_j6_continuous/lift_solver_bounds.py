"""In-place IK optimizer/constraint agreement for this profile only."""
from contextlib import contextmanager

@contextmanager
def synchronized_ik_bounds(solver):
    """Match optimizer position clamps to scoped checks, preserving CUDA pointers.

    Caller holds the Planner GPU lock and has already installed lift
    bounds. This context restores every touched tensor even on partial failure.
    """
    import torch
    safety = solver.solver.safety_rollout
    names = list(safety.joint_names)
    limits = safety.bound_constraint.joint_limits.position
    optimizers = solver.solver.optimizers
    if not optimizers:
        raise RuntimeError('GPU lift optimizers unavailable')
    saved = []

    def replace(tensor, value):
        if tensor.shape != value.shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid GPU optimizer lift bounds')
        saved.append((tensor, tensor.clone()))
        tensor.copy_(value)

    try:
        for opt in optimizers:
            order = list(opt.rollout_fn.joint_names)
            if len(order) != len(names) or set(order) != set(names):
                raise ValueError('GPU optimizer lift joint ordering mismatch')
            indices = [names.index(name) for name in order]
            low, high = opt.action_lows, opt.action_highs
            if low.shape != high.shape or not low.numel() or low.numel() % len(order):
                raise ValueError('GPU optimizer lift bound shape mismatch')
            repeat = low.numel() // len(order)
            wanted_low = limits[0, indices].repeat(repeat).reshape(low.shape)
            wanted_high = limits[1, indices].repeat(repeat).reshape(high.shape)
            # Never widen a pre-existing optimizer restriction.
            wanted_low = torch.maximum(low, wanted_low)
            wanted_high = torch.minimum(high, wanted_high)
            if bool((wanted_low >= wanted_high).any()):
                raise ValueError('Empty GPU optimizer lift bounds')
            replace(low, wanted_low)
            replace(high, wanted_high)
            if hasattr(opt, 'action_range'):
                replace(opt.action_range, high-low)
            # Keep Newton's numerical step preconditioner unchanged. CuRobo
            # scale_action uses a GLOBAL maximum over all DOFs; shrinking its
            # fixed-joint entries to ~1e-6 stalls even the free joints. Every
            # line-search candidate is still clamped by action_lows/highs.
        yield
    finally:
        for tensor, original in reversed(saved):
            tensor.copy_(original)

