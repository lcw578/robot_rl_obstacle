"""多番茄场景采样：单环境串行切换，不同时缓存多套整机。

训练集按打乱的循环均衡抽取，每回合可重新随机放置同一几何模型。
独立测试模型不进入训练调度。
"""
import numpy as np
import gymnasium as gym
from hopp import DegenerateHarvestSideError
from paper_method import HomeCollisionError, pose_reward


class ScenePoolEnv(gym.Wrapper):
    """只在reset时换整套植物几何和目标；同回合中绝不换目标。

    factory返回相同动作/观测空间的PaperPoseGoalEnv。
    HER奖励只读历史info，不读当前场景，因此换场景后仍能重标记旧样本。
    随机摆放造成HOME碰撞或式(4)选侧退化时重抽同一模型的位姿。
    """
    def __init__(self, env, factory, seeds, random_seed, *, randomize_pose=False):
        super().__init__(env)
        self.factory = factory
        self.scene_seeds = list(seeds)
        self.schedule_rng = np.random.default_rng(random_seed)
        self.randomize_pose = randomize_pose
        self.initial_collision_resamples = 0
        self.hopp_side_resamples = 0
        self.pending = []
        self.reward_batch_calls = 0

    @property
    def scene(self):
        return self.env.scene

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.schedule_rng=np.random.default_rng(seed);self.pending=[]
        if not self.pending:
            self.pending = self.schedule_rng.permutation(self.scene_seeds).tolist()
        chosen = int(self.pending.pop())
        if self.randomize_pose or chosen != self.env.scene.scene_seed:
            try:
                new = self._make_scene(chosen)
            except Exception as exc:
                raise RuntimeError(f'训练场景seed={chosen}构建/起点核对失败；未自动跳过：{exc}') from exc
            if new.observation_space!=self.observation_space or new.action_space!=self.action_space:
                new.close();raise ValueError('多场景动作/观测空间不一致')
            self.env.close();self.env=new
        return self.env.reset(seed=seed, options=options)

    def _make_scene(self, model_seed):
        if not self.randomize_pose:
            return self.factory(model_seed)
        pose_seeds = self.schedule_rng.integers(0, 2**63 - 1, size=32)
        last_error = None
        for seed in pose_seeds:
            try:
                return self.factory(model_seed, pose_seed=int(seed))
            except HomeCollisionError as exc:
                self.initial_collision_resamples += 1
                last_error = exc
            except DegenerateHarvestSideError as exc:
                self.hopp_side_resamples += 1
                last_error = exc
        raise RuntimeError('同一模型的32个随机位姿均被拒绝') from last_error

    def compute_reward(self,achieved_goal,desired_goal,info):
        if np.asarray(achieved_goal).ndim>1:self.reward_batch_calls+=1
        return pose_reward(achieved_goal,desired_goal,info)
