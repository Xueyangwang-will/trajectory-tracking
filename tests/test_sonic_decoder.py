"""Decoder state contract checks with the actual release model."""
from pathlib import Path
import unittest

import numpy as np

from trajectory_tracking.sonic_decoder import DEFAULT_JOINT_POS, RobotState, SonicDecoder

from trajectory_tracking.paths import ROOT, DEPLOY


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DEPLOY / 'policy/release/model_decoder.onnx').is_file():
            raise unittest.SkipTest('Download the SONIC release decoder model first')
        cls.decoder = SonicDecoder()

    def setUp(self):
        self.decoder.reset()
        self.state = RobotState(np.array([1., 0, 0, 0]), np.zeros(3),
                                DEFAULT_JOINT_POS.copy(), np.zeros(29))
        self.token = np.zeros(64, np.float32)

    def test_invalid_input_is_rejected_without_advancing_time(self):
        for invalid in [np.zeros(63), np.zeros((1, 64)), np.full(64, np.nan)]:
            with self.assertRaises(ValueError):
                self.decoder.step(invalid, self.state)
            self.assertEqual(self.decoder.ticks, 0)
        bad_state = RobotState(np.zeros(4), np.zeros(3), DEFAULT_JOINT_POS, np.zeros(29))
        with self.assertRaises(ValueError):
            self.decoder.step(self.token, bad_state)
        self.assertEqual(self.decoder.ticks, 0)

    def test_reset_restores_episode_start_and_output_does_not_alias_history(self):
        first = self.decoder.step(self.token, self.state)
        expected = first.action.copy()
        first.action[:] = 123  # Caller-owned output cannot alter previous-action memory.
        np.testing.assert_array_equal(self.decoder.last_action, expected)
        self.decoder.step(self.token, self.state)
        self.decoder.reset()
        again = self.decoder.step(self.token, self.state)
        np.testing.assert_array_equal(again.action, expected)
        self.assertEqual(self.decoder.ticks, 1)

    def test_root_translation_and_linear_velocity_are_not_decoder_inputs(self):
        a = np.r_[0., .35, .788740, 1., 0, 0, 0, DEFAULT_JOINT_POS]
        b = a.copy();b[:3] += [20., -10, .2]
        va = np.zeros(35);vb = va.copy();vb[:3] = [1., -2, .3]
        first = self.decoder.step(self.token, RobotState.from_mujoco(a, va))
        self.decoder.reset()
        second = self.decoder.step(self.token, RobotState.from_mujoco(b, vb))
        np.testing.assert_array_equal(first.observation, second.observation)
        np.testing.assert_array_equal(first.action, second.action)

    def test_failed_inference_does_not_commit_history(self):
        original = self.decoder.session
        class BrokenSession:
            def run(self, *_):
                raise RuntimeError("inference failure")
        self.decoder.session = BrokenSession()
        try:
            with self.assertRaises(RuntimeError):
                self.decoder.step(self.token, self.state)
            self.assertEqual(self.decoder.ticks, 0)
        finally:
            self.decoder.session = original
        self.decoder.step(self.token, self.state)
        self.assertEqual(self.decoder.ticks, 1)

    @unittest.skipUnless((ROOT / 'outputs/distinct_trajectories/trial-08/episode.npz').is_file(),
                         'Recorded trajectory fixture is not present')
    def test_chunked_stream_keeps_previous_actions(self):
        with np.load(ROOT / 'outputs/distinct_trajectories/trial-08/episode.npz') as source:
            q = source['qpos'][:20];v = source['qvel'][:20]
            token = source['token'][:20];expected = source['action'][:20]
        for chunk in [range(0, 7), range(7, 20)]:
            for k in chunk:
                out = self.decoder.step(token[k], RobotState.from_mujoco(q[k], v[k]))
                np.testing.assert_array_equal(out.action, expected[k])
        self.assertEqual(self.decoder.ticks, 20)


if __name__ == '__main__':
    unittest.main()
