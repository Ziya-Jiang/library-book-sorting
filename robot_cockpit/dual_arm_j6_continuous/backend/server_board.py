"""Opt-in server entry. Same controller owner/RT/watchdogs, local visual loop."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import asyncio
from . import controller_v31, server_v31
from .planner_bridge_board import PlannerBridge

# Changes apply only in this new process; server_v31 entry retains old bridge.
controller_v31.PlannerBridge=PlannerBridge
if __name__=='__main__':
    from .physical_profile import verify
    verify(server_v31.ROOT)
    asyncio.run(server_v31.main())
