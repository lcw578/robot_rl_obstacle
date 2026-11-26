"""论文公式与评估口径回归测试，无需训练或打开查看器。"""
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from env_cluster import _quat_to_mat
from evaluate_paper_dataset import evaluate_models
from hopp import DegenerateHarvestSideError, compute_hopp
from mocap_pose_controller import configure_mocap_xml
from paper_method import HAF_REFERENCE, HomeCollisionError, pose_reward
from paper_reward import terms
from paper_sac import fuse_tensor, latent_action_log_prob
from paper_scene_set import ScenePoolEnv


class PaperAlignmentTests(unittest.TestCase):
    def test_haf_uses_true_reference_and_short_arc_near_opposite(self):
        observation = torch.zeros((2, 19))
        observation[:, 3] = 1
        toward = torch.tensor([0., .001, -1.])
        obs = {'observation': observation, HAF_REFERENCE: toward.repeat(2, 1),
               'achieved_goal': torch.zeros((2, 6)),
               'desired_goal': torch.tensor([[0., 1., 0., 0., 1., 0.],
                                             [1., 0., 0., 1., 0., 0.]])}
        raw = torch.tensor([[0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., -1., 0., 0., 0.]])
        settings = {'haf': True, 'omega': 2., 'episode_steps': 50, 'time_index': 18,
                    'quaternion_parameterization': 'normalized'}
        actions = fuse_tensor(raw, obs, settings)
        torch.testing.assert_close(actions[0], actions[1])
        actual_z = _quat_to_mat(actions[0, 3:].numpy())[:, 2]
        np.testing.assert_allclose(actual_z, torch.nn.functional.normalize(toward, dim=0), atol=1e-6)

        half_turn = torch.tensor([[0., 0., 0., 0., 1., 0., 0.],
                                  [0., 0., 0., 0., -1., 0., 0.]])
        learned = fuse_tensor(half_turn, obs, {**settings, 'haf': False})
        torch.testing.assert_close(learned[0], learned[1])

    def test_first_step_entropy_uses_translation_marginal(self):
        observation = torch.zeros((2, 19), dtype=torch.float32)
        observation[1, 18] = 1 / 50
        settings = {'haf': True, 'omega': 2., 'episode_steps': 50, 'time_index': 18}
        component_log_prob = torch.tensor([[1., 2., 3., 4., 5., 6., 7.]]).repeat(2, 1)
        result = latent_action_log_prob(component_log_prob, observation, settings)
        torch.testing.assert_close(result, torch.tensor([6., 28.]))

    def test_near_open_boundary_scalar_and_her_reward_agree(self):
        desired = np.array([.5, .5, 0., 1., 0., 0.], dtype=np.float64)
        reached = desired.copy()
        reached[0] += 1e-9
        info = {'actual_joint_delta_norm_rad': .25, 'collision': False,
                'paper_reward_context': {'desired_goal': desired.tolist(), 'radius_m': .15,
                                         'hemisphere_normal': [1., 0., 0.],
                                         'zone_normal': [1., 0., 0.]}}
        detail = terms(reached, desired, info)
        self.assertTrue(detail['in_T'])
        self.assertEqual(pose_reward(reached, desired, info), np.float32(detail['total']))
        self.assertEqual(terms(desired, desired, info)['r_pos'], -10.)

    def test_only_known_pose_failures_are_resampled(self):
        kp = {'P0': [0., 0., 0.], 'M1': [0., 0., 1.], 'P3': [0., 1., 0.]}
        with self.assertRaises(DegenerateHarvestSideError):
            compute_hopp(kp, hemisphere_normal=(0., 1., 0.))
        failures = iter([DegenerateHarvestSideError(), HomeCollisionError(), None])
        expected = object()

        def factory(seed, **kwargs):
            self.assertEqual(seed, 42)
            error = next(failures)
            if error is not None:
                raise error
            return expected

        pool = SimpleNamespace(factory=factory, randomize_pose=True,
                               schedule_rng=np.random.default_rng(3),
                               initial_collision_resamples=0, hopp_side_resamples=0)
        self.assertIs(ScenePoolEnv._make_scene(pool, 42), expected)
        self.assertEqual((pool.hopp_side_resamples, pool.initial_collision_resamples), (1, 1))

    def test_explicit_false_disables_inherited_gravity_compensation(self):
        root = ET.fromstring('<mujoco><worldbody><body name="arm_base_mount">'
                             '<body name="arm" gravcomp="1"/></body></worldbody><actuator/></mujoco>')
        settings = dict(target_update='step_interpolated', physics_dt=.002,
                        weld_timeconst_s=.02, weld_torquescale_m=1.,
                        position_tolerance_m=.0005, rotation_tolerance_rad=.005,
                        safety_failure_penalty=0., gravity_compensation=False)
        configure_mocap_xml(root, {'pose_controller': settings, 'open_gear_rad': -.474},
                            np.zeros(3), np.eye(3))
        self.assertEqual(float(root.find(".//body[@name='arm']").get('gravcomp')), 0.)

    def test_path_excess_includes_failed_trajectories_with_sign(self):
        def factory(seed):
            if seed == 3:
                raise ValueError('invalid HOME')
            distance = .005 if seed == 1 else .10
            return SimpleNamespace(cfg={'max_episode_steps': 1}, close=lambda: None,
                                   reset=lambda **kwargs: (None, {'distance_m': .20, 'axis_error_rad': .2}),
                                   step=lambda action: (None, 0., True, False, {
                                       'distance_m': distance, 'axis_error_rad': .02,
                                       'collision': False, 'tcp_substep_path_length_m': .19 if seed == 1 else .09,
                                   }))

        report = evaluate_models(SimpleNamespace(predict=lambda *a, **k: (None, None)),
                                 factory, [1, 2, 3], {
                                     'position_tolerance_m': .015, 'axis_tolerance_rad': .05,
                                     'stagnation_distance_delta_m': .001, 'stagnation_angle_delta_rad': .001,
                                     'stagnation_consecutive_steps': 1, 'collision_retry': False,
                                 })
        self.assertEqual(report['success_rate'], 1/3)
        self.assertEqual(report['metrics']['path_excess_m']['count'], 2)
        self.assertAlmostEqual(report['metrics']['path_excess_m']['mean'], -.06)


if __name__ == '__main__':
    unittest.main()
