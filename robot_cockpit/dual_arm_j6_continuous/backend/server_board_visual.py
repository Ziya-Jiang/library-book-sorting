"""Existing robot owner plus locally leased visual worker; opt-in entry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import asyncio
from . import controller_v31,server_v31
from .planner_bridge_visual import PlannerBridge
from .startup_reservation import with_reservation
server_v31.Controller = with_reservation(server_v31.Controller)
controller_v31.PlannerBridge=PlannerBridge
if __name__=='__main__':asyncio.run(server_v31.main())
