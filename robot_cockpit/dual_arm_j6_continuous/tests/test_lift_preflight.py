"""Read-only preflight ordering, stop-on-failure and mutation boundaries."""
import sys,unittest,time,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lift_preflight import run_chain,validate_request,proxy_for

class LiftPreflightTests(unittest.TestCase):
    def model(self):
        def tcp(q):
            xyz=np.zeros((1,2,3));xyz[0,1]=np.asarray(q)[7:10]
            return xyz,np.tile([1.,0.,0.,0.],(1,2,1))
        return SimpleNamespace(tcp=tcp,bounds=np.array([[-3.]*14,[3.]*14]),
                               grip_matrix=np.eye(2),grip_bias=np.zeros(2),full_order=np.arange(16))
    def start(self):
        q=np.zeros(14);q[7:10]=[.35,-.2,-.01];return q
    def test_chained_start_is_previous_path_endpoint(self):
        model=self.model();q=self.start();seen=[]
        def plan(q,g,pos,quat,axis):
            seen.append((q.copy(),pos.copy(),axis));end=q.copy();end[7:10]=pos
            return end,dict(accepted=True)
        proof=run_chain(model,q,np.zeros(2),[.63,-.16,.04],50.,plan)
        self.assertTrue(proof['accepted']);self.assertTrue(proof['no_motion'])
        self.assertEqual([x[2] for x in seen],[1,0,2])
        np.testing.assert_allclose(seen[1][0][7:10],[.35,-.16,-.01])
        np.testing.assert_allclose(seen[2][0][7:10],[.63,-.16,-.01])
        np.testing.assert_allclose(seen[2][1],[.63,-.16,.04])
        np.testing.assert_allclose(q,self.start())
    def test_rejection_stops_before_following_phase(self):
        seen=[]
        def plan(q,g,pos,quat,axis):
            seen.append(axis);return None,dict(accepted=False,reason='collision_rejected')
        r=run_chain(self.model(),self.start(),np.zeros(2),[.63,-.16,.04],50.,plan)
        self.assertEqual(seen,[1]);self.assertEqual(r['failed_stage'],'approach_y')
        self.assertEqual(r['reason'],'collision_rejected');self.assertFalse(r['accepted'])
    def test_lift_failure_preserves_stage_and_reason(self):
        def plan(q,g,pos,quat,axis):
            if axis==2:return None,dict(accepted=False,reason='local_lift_ik_failed')
            end=q.copy();end[7:10]=pos;return end,dict(accepted=True)
        r=run_chain(self.model(),self.start(),np.zeros(2),[.63,-.16,.04],50.,plan)
        self.assertEqual(r['failed_stage'],'lift_z');self.assertEqual(len(r['phases']),3)
    def test_input_mismatch_and_out_of_bounds_are_not_clipped(self):
        request=dict(joints=self.start().tolist(),grippers=[0.,0.],target_xyz_m=[.63,-.16,.04],lift_mm=50.,deadline=time.monotonic()+30)
        validate_request(self.model(),request)
        for patch in [dict(lift_mm=40.),dict(lift_mm=float('nan')),dict(grippers=[-1.,0.]),dict(joints=[4.]*14),dict(target_xyz_m=[2.,-.16,.04])]:
            with self.subTest(patch=patch),self.assertRaises(ValueError):validate_request(self.model(),dict(request,**patch))
    def test_readonly_operation_dispatch_checks_identity(self):
        from gpu_preparation import PreparationService
        import lift_preflight
        service=PreparationService.__new__(PreparationService)
        service.identity=dict(planner_pid=123,planner_start='generation',revision='code')
        service.model=SimpleNamespace(digest='model')
        request=dict(service.identity,model_digest='model',operation=lift_preflight.OPERATION)
        with patch.object(lift_preflight,'solve',return_value={'preflight':{'accepted':False}}) as run:
            self.assertFalse(service.solve(request)['preflight']['accepted'])
            run.assert_called_once_with(service,request)
            with self.assertRaisesRegex(ValueError,'contract'):
                service.solve(dict(request,revision='old'))
            with self.assertRaisesRegex(ValueError,'Unknown'):
                service.solve(dict(request,operation='execute'))
            self.assertEqual(run.call_count,1)

    def test_proxy_has_no_command_io_and_does_not_alias_targets(self):
        model=self.model();q=self.start();original=dict(total_arm_joint_names=[str(i) for i in range(14)],
            target_link_names=['left','right'],arm_soft_joint_limits=model.bounds.copy(),
            _motion_gen=object(),tensor_args=object(),_v31_bridge=object(),
            target_arm_tool_position=np.ones((2,3)),target_arm_tool_orientation=np.ones((2,4)))
        p=SimpleNamespace(**original)
        scratch=proxy_for(p,model,q,np.zeros(2),np.array([.63,-.16,.04]),np.array([1.,0,0,0]),2,time.monotonic()+10)
        scratch.target_arm_tool_position[:]=0;scratch.arm_joint_position[:]=1;scratch.arm_soft_joint_limits[:]=0
        np.testing.assert_array_equal(p.target_arm_tool_position,np.ones((2,3)))
        np.testing.assert_array_equal(p.arm_soft_joint_limits,model.bounds)
        np.testing.assert_array_equal(q,self.start())
        for name in ['send','publish','curobo_cmd','_retrieve_motion_gen_result']:
            self.assertFalse(hasattr(scratch,name))

if __name__=='__main__':unittest.main()
