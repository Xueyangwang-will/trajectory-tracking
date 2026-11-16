import unittest
import numpy as np
from trajectory_tracking.collect import PathTracker, PrecisionTracker, sample
from trajectory_tracking.arm_export import soften,DEFAULT,ARM_JOINTS,LIMITS,RETAIN
from trajectory_tracking.generate_routes import resample,corner,hairpin


class TrackerTests(unittest.TestCase):
    def test_non_x_monotone_routes_and_rotated_terminal_heading(self):
        left=PrecisionTracker(path_xy=resample(corner(1)))
        self.assertAlmostEqual(left.goal_heading,np.pi/2)
        left.last_s=left.s[-1]-.14;left.filtered_closing_speed=.4
        result=left.command(50,left.path[-1]-[0,.14],np.pi/2,np.array([0,.4]))
        self.assertEqual(result[-1],3)
        back=PathTracker(path_xy=resample(hairpin()))
        self.assertAlmostEqual(abs(back.goal_heading),np.pi)
        self.assertTrue(np.any(np.diff(back.path[:,0])<0))

    def test_precision_terminal_motion_is_bounded_and_never_restarts_pursuit(self):
        tracker=PrecisionTracker()
        tracker.last_s=tracker.s[-1]-.5
        result=tracker.command(50,tracker.path[-1]-[.3,0],0.,np.zeros(2))
        self.assertEqual(result[0],.28)
        result=tracker.command(51,tracker.path[-1]-[.17,0],0.,np.zeros(2))
        self.assertEqual(result[-1],5)
        self.assertIsNotNone(tracker.arrived_since)
        result=tracker.command(52.1,tracker.path[-1]-[.17,0],0.,np.zeros(2))
        self.assertEqual(result[-1],3)
        result=tracker.command(53,tracker.path[-1]-[.4,0],0.,np.zeros(2))
        self.assertEqual(result[2],0)

    def test_precision_predicts_coast_and_stops_before_goal(self):
        tracker=PrecisionTracker()
        tracker.last_s=tracker.s[-1]-.14
        tracker.filtered_closing_speed=.4
        result=tracker.command(50,tracker.path[-1]-[.14,0],0.,np.array([.4,0]))
        self.assertEqual(result[-1],3)
        self.assertEqual(result[0],0)

    def test_offset_correction_and_rate_limits(self):
        tracker=PathTracker()
        for k in range(200):
            v,h,mode,e,s,point,phase=tracker.command(2+k*.02,np.array([0,.35]))
            self.assertLessEqual(v,.6+1e-9)
        self.assertLess(h,0)
        self.assertAlmostEqual(e,.35)
        self.assertAlmostEqual(np.linalg.norm(point-[.8,0]),0)

    def test_terminal_idle_is_latched(self):
        tracker=PathTracker()
        tracker.last_s=tracker.s[-1]-.1
        *_,phase=tracker.command(60,tracker.path[-1]-[.1,0])
        self.assertEqual(phase,3)
        v,h,mode,*_=tracker.command(61,tracker.path[-1]+[.4,.1])
        self.assertEqual(mode,0)

    def test_position_latch_before_heading_acceptance(self):
        tracker=PathTracker(arrival_radius=.30)
        tracker.last_s=tracker.s[-1]-.2
        command=tracker.command(60,tracker.path[-1]-[.2,0],np.deg2rad(40))
        self.assertEqual(command[-1],4)
        self.assertIsNotNone(tracker.arrived_since)
        # Drift outside 0.30 m cannot restart endpoint pursuit.
        command=tracker.command(61,tracker.path[-1]+[.5,.2],np.deg2rad(30))
        self.assertEqual(command[-1],4)
        self.assertAlmostEqual(tracker.goal_heading,0)
        command=tracker.command(62,tracker.path[-1]+[.5,.2],np.deg2rad(14))
        self.assertEqual(command[2],0)
        self.assertEqual(command[-1],3)

    def test_18cm_trigger_not_30cm(self):
        tracker=PathTracker(arrival_radius=.18)
        tracker.last_s=tracker.s[-1]-.25
        tracker.command(60,tracker.path[-1]-[.25,0],0.)
        self.assertIsNone(tracker.arrived_since)
        self.assertAlmostEqual(tracker.v,.15)
        tracker.command(61,tracker.path[-1]-[.17,0],0.)
        self.assertEqual(tracker.arrived_since,61)
        tracker.command(62,tracker.path[-1]-[.25,0],0.)
        self.assertEqual(tracker.idle_since,61)

    def test_cosine_braking(self):
        tracker=PathTracker()
        self.assertAlmostEqual(tracker.cosine_speed(1.2),.6)
        self.assertAlmostEqual(tracker.cosine_speed(.75),.375)
        self.assertAlmostEqual(tracker.cosine_speed(.3),.15)
        self.assertGreaterEqual(tracker.cosine_speed(-1),.15)

    def test_waypoints_use_30hz_terminal_token_and_world_height(self):
        tracker=PathTracker()
        pos,heading=tracker.waypoints(0,.6)
        np.testing.assert_allclose(pos[:,0],.6*np.arange(40,44)/30)
        np.testing.assert_allclose(pos[:,1],0)
        np.testing.assert_allclose(pos[:,2],.788740)
        np.testing.assert_allclose(heading,0)
        end,_=tracker.waypoints(tracker.s[-1]-.1,.15)
        np.testing.assert_allclose(end[:,:2],np.tile(tracker.path[-1],(4,1)))

    def test_arm_export_changes_only_arms_and_does_not_mutate_source(self):
        raw=np.zeros((5,36));raw[:,3]=1;raw[:,7:]=DEFAULT
        raw[:,7+ARM_JOINTS]+=np.linspace(-2,2,14)
        backup=raw.copy();out=soften(raw)
        np.testing.assert_array_equal(raw,backup)
        np.testing.assert_array_equal(out[:,:22],raw[:,:22])
        self.assertTrue(np.all(abs(out[:,7+ARM_JOINTS]-DEFAULT[ARM_JOINTS])<=RETAIN*LIMITS+1e-12))

    def test_quaternion_sampling_handles_sign_and_endpoint(self):
        f=np.zeros((2,36));f[0,3]=1;f[1,3]=-1
        out=sample(f,np.array([0,.5,1,2]))
        np.testing.assert_allclose(np.linalg.norm(out[:,3:7],axis=1),1)
        np.testing.assert_allclose(abs(out[:,3]),1)


if __name__=='__main__':unittest.main()
