import sys, unittest, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from preparation_connection import check_path,bind_start_and_goal
import gpu_preparation as gpu
import j6_geometry as geo
from backend.board_servo.model import Model

class ConnectionChecks(unittest.TestCase):
    def test_feedback_tick_difference_rebinds_only_inactive_left_goal(self):
        incoming=np.zeros(14);measured=incoming.copy();measured[:7]=[0,.000024,-.000025,0,0,0,.000049]
        measured[9]=.0006
        requested=incoming.copy();requested[7:]=np.arange(7)*.1
        original=requested.copy()
        q,goal,detail=bind_start_and_goal(incoming,measured,requested,[False,True])
        np.testing.assert_array_equal(q,measured)
        np.testing.assert_array_equal(goal[:7],measured[:7])
        np.testing.assert_array_equal(goal[7:],requested[7:])
        np.testing.assert_array_equal(requested,original)
        self.assertAlmostEqual(detail['max_inactive_left_delta_rad'],.000049)
        with self.assertRaisesRegex(ValueError,'left'):
            check_path(np.array([q,requested]),q,np.array([[-2.]*14,[2.]*14]))
        check_path(np.array([q,goal]),q,np.array([[-2.]*14,[2.]*14]))
    def test_rebinding_does_not_hide_owner_intent_or_large_start_change(self):
        q=np.zeros(14);goal=q.copy()
        with self.assertRaisesRegex(ValueError,'enable left'):bind_start_and_goal(q,q,goal,[True,True])
        goal[0]=.003
        with self.assertRaisesRegex(ValueError,'inactive left'):bind_start_and_goal(q,q,goal,[False,True])
        goal[0]=0.;changed=q.copy();changed[10]=.003
        with self.assertRaisesRegex(ValueError,'input start'):bind_start_and_goal(q,changed,goal,[False,True])
    def test_rebound_path_still_rejects_small_commanded_left_motion(self):
        q,goal,_=bind_start_and_goal(np.zeros(14),np.full(14,.000024),np.zeros(14),[False,True])
        points=np.array([q,goal]);points[-1,0]+=.000024
        with self.assertRaisesRegex(ValueError,'left'):check_path(points,q,np.array([[-2.]*14,[2.]*14]))
    def test_actual_single_joint_sweep_keeps_other_thirteen_fixed(self):
        from j6_trajectory import points_for,check_samples
        from j6_geometry import SWEEP_DEG
        q=np.zeros(14);goal=q.copy();goal[12]=np.deg2rad(SWEEP_DEG)
        points=points_for(q,goal,12,20)
        np.testing.assert_array_equal(points[:,np.arange(14)!=12],0.)
        check_samples(points,q,12,np.array([[-2.]*14,[2.]*14]),exact_fixed=True)
        goal[13]=.1
        with self.assertRaisesRegex(ValueError,'other joints'):points_for(q,goal,12,20)
    def test_free_j7_still_holds_left_and_position_limits(self):
        q=np.zeros(14);b=np.array([[-2.]*14,[2.]*14]);a=np.zeros((3,14))
        a[1,13]=1.;check_path(a,q,b)
        a[1,0]=.003
        with self.assertRaisesRegex(ValueError,'left'):check_path(a,q,b)
        a[1,0]=0.;a[1,13]=3.
        with self.assertRaisesRegex(ValueError,'limit'):check_path(a,q,b)
    def test_geometry_pass_does_not_accept_failed_connection(self):
        m=Model(ROOT/'backend/board_servo')
        q=np.asarray(json.loads((ROOT/'tests/fixtures/preparation_194325.json').read_text())['start_joints_rad'])
        goal=q.copy();goal[13]+=.1
        xyz,quat=m.tcp(goal)
        # Only isolate geometry for this control-flow regression. The dedicated
        # axis/FK and path tests verify those computations separately.
        calls=[]
        def solve(*a,connection_goal=None,**kw):
            if connection_goal is not None:
                calls.append(connection_goal.copy());return dict(accepted=False,reason='collision')
            return [goal.tolist()],dict()
        with patch.object(gpu,'solve_candidates',side_effect=solve),patch.object(geo,'axis_error',return_value=0.),patch.object(geo,'collision_free',return_value=True),patch.object(geo,'j7_align_goal',return_value=(goal,{})),patch.object(geo,'rotation') as rot:
            rot.return_value.apply.return_value=np.array([1.,0.,0.])
            jobs=[(1.,m.bounds[0,7:],m.bounds[1,7:],q[7:],'test',0.,0.,quat[0,1])]
            with patch.object(geo,'preparation_jobs',return_value=iter(jobs)):
                with self.assertRaises(geo.AlignmentRejected) as error:
                    geo.align_goal(m,q,[0.,0.],xyz[0,1],search_seconds=1.)
            self.assertEqual(len(calls),1)
            self.assertIn('connection_path',error.exception.summary['best_candidates'][0]['rejection_reasons'])

if __name__=='__main__':unittest.main()
