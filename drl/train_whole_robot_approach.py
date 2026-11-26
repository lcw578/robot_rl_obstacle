"""训练 AUBO i5 的论文仿真路线（MuJoCo mocap、HAF、HER-SAC）。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from project_paths import CONFIG_DIR, DEFAULT_CONFIG, repo_relative

GUARD = 'TOMATO_PAPER_GUARDED'


def run(args, config, output):
    import torch
    import yaml
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from paper_dataset import read_manifest
    from paper_method import PaperPoseGoalEnv, fusion_settings_from_config
    from paper_sac import PaperHerReplayBuffer, PaperSAC, PaperSACPolicy, evaluate
    from paper_scene_set import ScenePoolEnv
    from robot_model import MODEL

    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前 Python 环境未检测到可用 CUDA')

    manifest_path, dataset = read_manifest(config)
    config['paper_training']['dataset_manifest'] = repo_relative(manifest_path)
    config['paper_training']['dataset_manifest_sha256'] = digest(manifest_path)
    training_seeds = dataset['training_seeds']
    anchor_seed = config['scene_seed']
    if anchor_seed not in training_seeds:
        raise ValueError('训练锚点必须属于 400 个训练模型')
    fusion = fusion_settings_from_config(config)

    def make_env(seed, snapshot=None, *, pose_seed=None):
        return PaperPoseGoalEnv(seed, config=config, visuals=False, robot_snapshot=snapshot,
                                pose_seed=pose_seed)

    anchor = make_env(anchor_seed)
    env = anchor
    try:
        snapshot = output / 'scene'
        anchor.scene.export(snapshot)
        (output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        geometry_files = [MODEL, *sorted(CONFIG_DIR.glob('*.yaml'))]
        geometry_files = [file for file in geometry_files if file.name != DEFAULT_CONFIG.name]
        geometry_hashes = {file.name: digest(file) for file in geometry_files}
        (output / 'geometry_sha256.json').write_text(json.dumps(geometry_hashes, indent=2))

        class StopAfterEpisode(BaseCallback):
            def __init__(self):
                super().__init__()
                self.stop_after_rollout = False
                self.episodes = 0

            def _on_step(self):
                if self.locals['dones'][0]:
                    self.episodes += 1
                self.stop_after_rollout = (
                    (args.episodes is not None and self.episodes >= args.episodes)
                    or (output / 'STOP').exists()
                )
                return True

        from paper_sac import evaluation_clone
        eval_env = evaluation_clone(anchor)
        try:
            from functools import partial
            pool = ScenePoolEnv(anchor, partial(make_env, snapshot=snapshot), training_seeds, args.seed,
                                randomize_pose='rigid_rotation_deg' in config['paper_training'])
            env = Monitor(pool, str(output / 'monitor.csv'))
            model = PaperSAC(
                PaperSACPolicy, env, device=args.device, seed=args.seed,
                learning_rate=3e-4, learning_starts=max(256, config['max_episode_steps']),
                batch_size=256, buffer_size=args.buffer_size, gamma=.99, tau=.005,
                ent_coef='auto_0.1', target_entropy=-1.,
                train_freq=1, gradient_steps=1,
                policy_kwargs={'net_arch': [128, 128], 'fusion_settings': fusion},
                replay_buffer_class=PaperHerReplayBuffer,
                replay_buffer_kwargs={'n_sampled_goal': 4, 'goal_selection_strategy': 'future', 'copy_info_dict': True},
            )
            stop = StopAfterEpisode()
            model.learn(total_timesteps=args.steps, callback=stop, log_interval=10)
            model.save(output / 'model.zip')
            model.save_replay_buffer(output / 'replay_buffer.pkl')
            result = evaluate(model, eval_env)
            report = {
                'algorithm': 'SAC + future-HER (k=4) + HAF',
                'action_contract': fusion['action_contract'],
                'entropy_definition': 'latent tanh-Gaussian, translation only when HAF gain is exactly one',
                'target_entropy': model.target_entropy,
                'replay_buffer_size': model.buffer_size,
                'execution': 'mocap_weld',
                'dataset_manifest': config['paper_training']['dataset_manifest'],
                'dataset_manifest_sha256': config['paper_training']['dataset_manifest_sha256'],
                'dataset_provenance': dataset.get('provenance'),
                'robot_model_sha256': geometry_hashes[MODEL.name],
                'robot_adaptation': {
                    'robot': 'AUBO i5',
                    'home_joints_deg': config['start_joints_deg'],
                    'base_yaw_offset_deg': config.get('base_yaw_offset_deg', 0.),
                    'tcp': anchor.scene.manifest['tcp'],
                    'gripper_mass_kg': anchor.scene.manifest['gripper_mass_kg'],
                    'target_workspace': config.get('target_workspace'),
                    'pose_controller': config['pose_controller'],
                    'control_dt': config['control_dt'],
                },
                'robot_snapshot': 'scene/scene.json',
                'effective_config': 'config.yaml',
                'training_scene_count': len(training_seeds),
                'held_out_scene_count': len(dataset['test_seeds']),
                'initial_collision_resamples': pool.initial_collision_resamples,
                'hopp_side_resamples': pool.hopp_side_resamples,
                'rigid_rotation_deg': config['paper_training'].get('rigid_rotation_deg'),
                'seed': args.seed,
                'device': str(model.device),
                'requested_steps': args.steps,
                'requested_episodes': args.episodes,
                'steps': model.num_timesteps,
                'episodes': stop.episodes,
                'her_reward_batches': pool.reward_batch_calls,
                'anchor_evaluation': result,
            }
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print('训练与锚点评估已写入', output, flush=True)
        finally:
            eval_env.close()
    finally:
        env.close()


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000, help='本次采样步数上限')
    parser.add_argument('--episodes', type=int, help='本次完整回合数上限')
    parser.add_argument('--buffer-size', type=int)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--memory-budget-mb', type=int, default=4096)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text())
    if args.buffer_size is None:
        args.buffer_size = config['paper_training'].get('replay_buffer_size', 20000)
    config['paper_training']['replay_buffer_size'] = args.buffer_size
    if config['pose_controller']['mode'] != 'mocap_weld':
        parser.error('仿真执行器须为 mocap_weld')
    horizon = config['max_episode_steps']
    if type(horizon) is not int or horizon < 1:
        parser.error('max_episode_steps 须为正整数')
    if args.steps < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error('步数和回合预算须为正整数')
    if args.buffer_size <= max(horizon, 256) or args.memory_budget_mb < 1:
        parser.error('经验池须大于回合长度和 256 条样本，内存预算须为正整数')

    output = args.output.resolve()
    if os.environ.get(GUARD) == '1':
        run(args, config, output)
        return 0

    from resource_guard import guarded_run, startup_memory_message
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, GUARD: '1'}
    result = guarded_run(
        [sys.executable, '-u', __file__, *sys.argv[1:]], None, Path.cwd(), environment,
        timeout=float('inf'), memory_mb=args.memory_budget_mb, reserve_mb=1024,
        cores=2, address_space_mb='inherit', stop_file=output / 'STOP',
    )
    (output / 'resource_report.json').write_text(json.dumps(result, indent=2))
    print(startup_memory_message(result) or result, flush=True)
    return result['exit_code'] if result['resource_status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
