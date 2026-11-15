"""Stage identity and budget regressions; never construct robot transports."""
import sys,copy,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import preparation_target as target
import j6_geometry as geo
import gpu_preparation as gpu
from lift_preflight import run_chain


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.q=np.zeros(14);self.q[7:10]=[.35,-.2,-.01]
        def tcp(q):
            xyz=np.zeros((1,2,3));xyz[0,1]=np.asarray(q)[7:10]
            return xyz,np.tile([1.,0.,0.,0.],(1,2,1))
        self.model=SimpleNamespace(tcp=tcp,bounds=np.array([[-3.]*14,[3.]*14]),
                                   offsets=np.array([[1.,0.,0.],[1.,0.,0.]]))
        self.goal=np.array([.70,-.25,.03]);self.trace=[]
        self.ref=dict(batch_id='this-run',epoch=2,frame_token='frame',
            method='initial_left_wrist_spine_base_Z_median',
            far_edge=dict(reference='top_book_farthest_edge_1_2'),
            contact_y_window=dict(available=True,lower_y_mm=-260.,upper_y_mm=-240.,
                                 source='same_frame_spine_and_far_edge_1_2_base_Y'))

    def predict(self,point,seconds):
        self.trace.append(('predict',point.copy(),seconds))
        def plan(q,grip,pos,quat,axis):
            self.trace.append(('phase',axis,q.copy()))
            end=q.copy();end[7:10]=pos;end[10]+=.1
            return end,dict(accepted=True,end_joints_rad=end.tolist())
        return run_chain(self.model,self.q,np.zeros(2),point,40.,plan)

    def geometry(self):
        return dict(position_error_mm=.001,axis_error_deg=.001,tool_forward_error_deg=.001)

    def select(self,**kwargs):
        return target.select_preparation(self.model,self.q,[0.,0.],self.goal,self.ref,'this-run',2,
                                         predict_path=kwargs.pop('predict_path',self.predict),**kwargs)

    def test_connection_uses_lift_endpoint_and_does_not_change_original_start(self):
        initial=self.q.copy()
        def align(model,q,grip,**kw):
            self.trace.append(('align',q.copy()))
            np.testing.assert_allclose(q[7:10],self.goal)
            self.assertAlmostEqual(q[10],.3)
            self.assertNotIn('target_position_m',kw)
            self.assertGreaterEqual(kw['search_seconds'],2.)
            return q.copy(),self.geometry()
        with patch.object(target,'align_goal',side_effect=align):goal,detail=self.select()
        self.assertEqual([x[0] for x in self.trace],['predict','phase','phase','phase','align'])
        self.assertEqual([x[1] for x in self.trace if x[0]=='phase'],[1,0,2])
        np.testing.assert_array_equal(self.q,initial)
        np.testing.assert_allclose(goal,detail['predicted_lift_joints_rad'])
        self.assertEqual(detail['preparation_start_source'],'validated_Y_X_lift_endpoint')

    def test_each_y_candidate_gets_its_own_prediction(self):
        seen=[]
        def align(model,q,grip,**kw):
            seen.append(q.copy())
            if len(seen)==1:
                raise geo.AlignmentRejected(dict(best_candidates=[dict(rejection_reasons=['connection_path'])],budget_exhausted=False))
            return q.copy(),self.geometry()
        with patch.object(target,'align_goal',side_effect=align):_,detail=self.select()
        self.assertEqual(len(seen),2)
        self.assertNotEqual(seen[0][8],seen[1][8])
        predictions=[x[1] for x in self.trace if x[0]=='predict']
        self.assertEqual(len(predictions),2)
        np.testing.assert_allclose(predictions[1],seen[1][7:10])
        self.assertAlmostEqual(detail['selected_target_mm'][1],seen[1][8]*1000)

    def test_failed_path_never_enters_preparation_and_keeps_real_reason(self):
        def fail(point,seconds):
            return dict(accepted=False,failed_stage='advance_x',reason='collision_path_rejected')
        with patch.object(target,'align_goal') as align,self.assertRaises(target.PreparationRejected) as e:
            self.select(predict_path=fail)
        align.assert_not_called()
        self.assertIn('advance_x',str(e.exception))
        self.assertEqual(e.exception.preparation_diagnostic['attempts'][0]['reason'],'collision_path_rejected')

    def test_reject_wrong_candidate_or_partial_prediction(self):
        proof=self.predict(self.goal,10.)
        changes=[dict(target_xyz_m=[.70,-.24,.03]),dict(end_joints_rad=[4.]*14),
                 dict(start_joints_rad=(self.q+.01).tolist()),dict(lift_mm=30.),dict(phases=[])]
        changed=copy.deepcopy(proof);changed['end_joints_rad'][0]+=.001;changes.append(changed)
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                target.predicted_lift_state(self.model,dict(proof,**change),self.q,self.goal,40.)

    def test_late_preparation_success_is_not_accepted(self):
        clock=[100.]
        def align(model,q,grip,**kw):
            clock[0]+=46.;return q.copy(),self.geometry()
        with patch.object(target.time,'monotonic',side_effect=lambda:clock[0]),patch.object(target,'align_goal',side_effect=align):
            with self.assertRaises(target.PreparationRejected) as e:self.select()
        self.assertEqual(e.exception.preparation_diagnostic['attempts'][0]['reason'],'late_preparation_result_discarded')

    @staticmethod
    def axis_failure(y,error,**checks):
        valid=dict(position=True,tool_forward=True,joint_limits=True,held_left=True,
                   endpoint_collision_free=True,axis=error<=1.)
        valid.update(checks)
        return dict(y_mm=y,failed_stage='preparation_geometry',detail=dict(best_candidates=[dict(
            axis_error_deg=error,limits=dict(axis_deg=1.),checks=valid,rejection_reasons=['axis'])]))

    def test_first_complete_candidate_returns_without_extra_margin_search(self):
        seen=[]
        def align(model,q,grip,**kw):
            seen.append(float(q[8]))
            if len(seen)==1:raise geo.AlignmentRejected(self.axis_failure(-250.,3.)['detail'])
            # Within existing .5 mm / 1 degree hard limits, outside the old
            # .25 mm / .5 degree optional preference. No third call allowed.
            if len(seen)>2:raise AssertionError('Already accepted; must not search again')
            return q.copy(),dict(position_error_mm=.4,axis_error_deg=.8,tool_forward_error_deg=.7)
        with patch.object(target,'align_goal',side_effect=align):_,detail=self.select()
        self.assertEqual(len(seen),2)
        self.assertGreater(detail['attempts'][-1]['quality_score'],1.)
        self.assertEqual(detail['selection_policy'],'first_fully_valid_candidate')

    def test_axis_trend_can_prioritize_either_direction(self):
        for sign in (-1.,1.):
            with self.subTest(sign=sign):
                observations=[self.axis_failure(-200.,3.2),self.axis_failure(-200.+sign*10,1.8)]
                pending=[-200.-sign*10,-200.+sign*20,-200.+sign*30]
                y,why=target.next_contact_candidate(pending,-200.,observations)
                self.assertEqual(y,-200.+sign*20)
                self.assertEqual(why['policy'],'same_run_axis_trend_one_step')
                self.assertLessEqual(abs(why['extrapolation_step_mm']),10.)
                self.assertEqual(len(pending),3)  # sorting never drops candidates

    def test_collision_or_limit_rejected_geometry_does_not_steer_search(self):
        for field in ('joint_limits','endpoint_collision_free','held_left','position','tool_forward'):
            with self.subTest(field=field):
                observations=[self.axis_failure(-200.,3.2),self.axis_failure(-210.,.1,**{field:False})]
                y,why=target.next_contact_candidate([-190.,-220.],-200.,observations)
                self.assertEqual(y,-190.)
                self.assertEqual(why['policy'],'nearest_center')

    def test_missing_or_nonfinite_diagnostics_keep_nearest_center_fallback(self):
        for bad in ({},self.axis_failure(-210.,float('nan')),self.axis_failure(-210.,float('inf'))):
            with self.subTest(bad=bad):
                y,why=target.next_contact_candidate([-190.,-220.],-200.,[self.axis_failure(-200.,3.2),bad])
                self.assertEqual(y,-190.)
                self.assertEqual(why['policy'],'nearest_center')

    def test_trend_never_accepts_geometry_with_failed_connection(self):
        self.ref['contact_y_window'].update(lower_y_mm=-290.,upper_y_mm=-210.)
        seen=[]
        def align(model,q,grip,**kw):
            y=round(float(q[8])*1000,6);seen.append(y)
            if y==-280.:return q.copy(),self.geometry()
            error={-250.:3.2,-260.:1.8,-270.:.8}.get(y,5.)
            detail=self.axis_failure(y,error)['detail']
            if y==-270.:detail['best_candidates'][0]['rejection_reasons']=['connection_path']
            raise geo.AlignmentRejected(detail)
        with patch.object(target,'align_goal',side_effect=align):_,detail=self.select()
        self.assertEqual(seen,[-250.,-260.,-270.,-280.])
        self.assertEqual(detail['attempts'][2]['failed_stage'],'preparation_connection')
        predictions=[row[1][1]*1000 for row in self.trace if row[0]=='predict']
        np.testing.assert_allclose(predictions,seen)

    def alignment_context(self,seconds=8.,connection_delay=.2):
        from contextlib import ExitStack
        stack=ExitStack();clock=[100.];calls=[];q=self.q.copy();goal=q.copy();goal[13]=.1
        def solve(*args,connection_goal=None,**kw):
            calls.append((connection_goal,args[-1]))
            if connection_goal is not None:
                clock[0]+=connection_delay;return dict(accepted=True)
            clock[0]+=1.2;return [goal.tolist(),goal.tolist()],dict()
        stack.enter_context(patch.object(geo.time,'monotonic',side_effect=lambda:clock[0]))
        stack.enter_context(patch.object(gpu,'solve_candidates',side_effect=solve))
        stack.enter_context(patch.object(geo,'axis_base',return_value=np.array([0.,1.,0.])))
        stack.enter_context(patch.object(geo,'axis_error',return_value=0.))
        stack.enter_context(patch.object(geo,'collision_free',return_value=True))
        stack.enter_context(patch.object(geo,'j7_align_goal',return_value=(goal,{})))
        jobs=[(1.,self.model.bounds[0,7:],self.model.bounds[1,7:],q[7:],'offline',0.,0.,np.array([1.,0.,0.,0.]))]
        stack.enter_context(patch.object(geo,'preparation_jobs',return_value=iter(jobs)))
        return stack,q,calls

    def test_ik_over_one_second_still_has_separate_connection_time(self):
        ctx,q,calls=self.alignment_context()
        with ctx:_,detail=geo.align_goal(self.model,q,[0.,0.])
        self.assertEqual(len(calls),2)  # duplicate seeds checked only once
        self.assertGreater(calls[1][1],1.)
        self.assertEqual(detail['budgets']['connection_reserved_s'],4.)

    def test_late_connection_proof_is_discarded(self):
        ctx,q,calls=self.alignment_context(connection_delay=9.)
        with ctx,self.assertRaises(geo.AlignmentRejected) as e:geo.align_goal(self.model,q,[0.,0.])
        self.assertIn('connection_deadline',e.exception.summary['best_candidates'][0]['rejection_reasons'])


if __name__=='__main__':unittest.main()
