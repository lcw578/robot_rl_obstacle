"""检查 AUBO i5 论文仿真的场景、观测接口与 mocap 响应。"""
import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='数字模型编号，默认使用配置中的锚点')
    parser.add_argument('--view', action='store_true', help='打开 MuJoCo 窗口')
    args = parser.parse_args()

    import numpy as np
    from paper_method import PaperPoseGoalEnv

    env = PaperPoseGoalEnv(args.seed, render_mode='human' if args.view else None,
                           visuals=args.view)
    try:
        observation, start = env.reset()
        action = np.array([.02, 0., 0., 1., 0., 0., 0.], dtype=np.float32)
        _, reward, terminated, truncated, feedback = env.step(action)
        print(json.dumps({
            'scene_seed': env.scene.scene_seed,
            'arm_joints': len(env.scene.arm_jids),
            'action_shape': env.action_space.shape,
            'observations': {key: value.shape for key, value in observation.items()},
            'initial_distance_m': start['distance_m'],
            'execution': feedback['execution'],
            'requested_translation_m': feedback['requested_translation_norm_m'],
            'tracking_error_m': feedback['pose_tracking_distance_m'],
            'collision': feedback['collision'],
            'execution_failure': feedback.get('execution_failure_reason'),
            'terminated': terminated,
            'truncated': truncated,
            'reward': reward,
        }, ensure_ascii=False, indent=2))
        if args.view:
            while env.viewer.is_running():
                env.render()
                time.sleep(.05)
        return int(feedback['collision'] or bool(feedback.get('execution_failure')))
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
