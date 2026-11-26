"""Stable-Baselines3 的 SAC、HAF 动作融合和位姿 future-HER。"""
import copy
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.sac.policies import Actor, MultiInputPolicy

from paper_method import (HAF_REFERENCE, PAPER_DISTANCE, PAPER_NORMAL, PAPER_OBS_SIZE,
                          QUAT, TIME, approach_motion_summary)
from paper_method import reward_replay_info, shift_her_goals


def haf_gain(state, settings):
    index = settings.get('time_index', TIME)
    step = th.round(state[..., index:index+1] * settings['episode_steps'])
    return th.exp(-settings['omega'] * step) if settings['haf'] else th.zeros_like(step)


def canonical_quaternion(q):
    """用最大绝对值分量的符号确定唯一表示，半圈旋转也适用。"""
    leading = q.gather(-1, q.abs().argmax(dim=-1, keepdim=True))
    return th.where(leading < 0, -q, q)


def fuse_tensor(raw, obs, settings):
    """共享式 (21)、(23)–(25)，使用实际场景参考点和整数决策步。"""
    state = obs['observation']
    w, x, y, z = F.normalize(state[..., QUAT], dim=-1).unbind(-1)
    tool_z = th.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)
    delta = (obs[HAF_REFERENCE] - obs['achieved_goal'][..., :3]).to(tool_z.dtype)
    norm = th.linalg.vector_norm(delta, dim=-1, keepdim=True)
    toward = delta / norm.clamp_min(1e-8)
    dot = (tool_z * toward).sum(-1, keepdim=True).clamp(-1, 1)
    cross = th.cross(tool_z, toward, dim=-1)
    sine = th.linalg.vector_norm(cross, dim=-1, keepdim=True)
    angle = th.atan2(sine, dot) / 2
    h = th.cat((th.cos(angle), th.sin(angle) * cross / sine.clamp_min(1e-7)), dim=-1)
    basis = F.one_hot(tool_z.abs().argmin(-1), 3).to(tool_z.dtype)
    opposite_axis = F.normalize(th.cross(tool_z, basis, dim=-1), dim=-1)
    opposite = th.cat((th.zeros_like(dot), opposite_axis), dim=-1)
    identity = th.zeros_like(h)
    identity[..., 0] = 1
    h = th.where(sine < 1e-7, th.where(dot < 0, opposite, identity), h)
    h = canonical_quaternion(th.where(norm < 1e-8, identity, h))

    policy_quaternion = raw[..., 3:]
    if settings.get('quaternion_parameterization') != 'normalized':
        policy_quaternion = identity + settings['quaternion_scale'] * policy_quaternion
    length = th.linalg.vector_norm(policy_quaternion, dim=-1, keepdim=True)
    policy = th.where(length >= 1e-8, policy_quaternion / length.clamp_min(1e-8), identity)
    policy = canonical_quaternion(policy)
    alpha = haf_gain(state, settings)
    dot = (policy * h).sum(-1, keepdim=True)
    short_arc_h = th.where(dot < 0, -h, h)
    dot = dot.abs()
    theta = th.acos(dot.clamp(max=.999999))
    curved = (th.sin((1-alpha)*theta)*policy + th.sin(alpha*theta)*short_arc_h) / th.sin(theta)
    linear = (1-alpha)*policy + alpha*short_arc_h
    fused = F.normalize(th.where(dot > .9995, linear, curved), dim=-1)
    fused = th.where(alpha == 1, h, th.where(alpha == 0, policy, fused))
    return th.cat((raw[..., :3], canonical_quaternion(fused)), dim=-1)


def latent_action_log_prob(component_log_prob, state, settings):
    """平移始终有效，启发式完全接管旋转时省略旋转潜变量的熵。"""
    translation = component_log_prob[..., :3].sum(-1)
    rotation = component_log_prob[..., 3:].sum(-1)
    return translation + th.where(haf_gain(state, settings).squeeze(-1) == 1, 0., rotation)


class FusionActor(Actor):
    def __init__(self,*args,fusion_settings,**kwargs):
        super().__init__(*args,**kwargs);self.fusion_settings=fusion_settings
        if not isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            raise ValueError('HAF策略使用对角tanh高斯分布')

    def forward(self,obs,deterministic=False):
        return fuse_tensor(super().forward(obs,deterministic),obs,self.fusion_settings)

    def action_log_prob(self,obs):
        mean, log_std, kwargs = self.get_action_dist_params(obs)
        raw = self.action_dist.actions_from_params(mean, log_std, **kwargs)
        # 归一化与SLERP的映射非一一对应。旋转熵定义在融合前的潜变量上。
        gaussian = self.action_dist.distribution.log_prob(self.action_dist.gaussian_actions)
        component_log_prob = gaussian - th.log(1 - raw.square() + self.action_dist.epsilon)
        log_prob = latent_action_log_prob(component_log_prob, obs['observation'], self.fusion_settings)
        return fuse_tensor(raw, obs, self.fusion_settings), log_prob


class PaperSACPolicy(MultiInputPolicy):
    def __init__(self,*args,fusion_settings,**kwargs):
        self.fusion_settings=fusion_settings
        super().__init__(*args,**kwargs)

    def make_actor(self,features_extractor=None):
        kwargs=self._update_features_extractor(self.actor_kwargs,features_extractor)
        return FusionActor(**kwargs,fusion_settings=self.fusion_settings).to(self.device)

    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters();data['fusion_settings']=self.fusion_settings
        return data


class PaperSAC(SAC):
    def collect_rollouts(self, env, callback, *args, **kwargs):
        # 先保存本步转移，再响应回合预算或 STOP。
        result = super().collect_rollouts(env, callback, *args, **kwargs)
        if getattr(callback, 'stop_after_rollout', False):
            return result._replace(continue_training=False)
        return result

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if action_noise is not None:
            raise ValueError('HAF使用策略采样探索，不支持融合后叠加动作噪声')
        actions, _ = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.num_timesteps < learning_starts:
            with th.no_grad():
                actions = fuse_tensor(th.as_tensor(actions, device=self.device),
                                      obs_as_tensor(self._last_obs, self.device),
                                      self.policy.fusion_settings).cpu().numpy()
        # 环境动作已在 [-1, 1] 内，经验池存入同一命令以避免重新缩放的舍入误差。
        return actions, actions.copy()


class PaperHerReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, interior_offset_m=1e-5, **kwargs):
        self.interior_offset_m = interior_offset_m
        super().__init__(*args, **kwargs)

    def add(self, obs, next_obs, action, reward, done, infos):
        compact = [reward_replay_info(info) for info in infos]
        return super().add(obs, next_obs, action, reward, done, compact)

    def _sample_goals(self, batch_indices, env_indices):
        goals = super()._sample_goals(batch_indices, env_indices)
        return shift_her_goals(goals, self.infos[batch_indices, env_indices], self.interior_offset_m)

    def _normalize_obs(self, obs, env=None):
        # 同步式(14)中可重标记的字段，HAF_REFERENCE保留真实场景的位置。
        state = obs['observation']
        if state.shape[-1] == PAPER_OBS_SIZE:
            goal = obs['desired_goal']
            state = state.copy()
            state[..., PAPER_DISTANCE] = np.linalg.norm(obs['achieved_goal'][..., :3] - goal[..., :3], axis=-1)
            state[..., PAPER_NORMAL] = goal[..., 3:]
            obs = {**obs, 'observation': state}
        return super()._normalize_obs(obs, env)


def evaluate(model,env):
    started=time.perf_counter()
    obs,initial=env.reset(seed=123);rows=[];minimum=initial['distance_m'];total=0.
    start_position=env.scene.tcp_pose()[0].copy()
    start_joints=env.data.qpos[env.scene.arm_qadr].copy()
    for step in range(env.cfg['max_episode_steps']):
        action=model.predict(obs,deterministic=True)[0]
        obs,reward,term,trunc,info=env.step(action);minimum=min(minimum,info['distance_m']);total+=reward
        rows.append({'step':step+1,'reward':reward,'arm_rad':env.data.qpos[env.scene.arm_qadr].tolist(),
            'action':action.tolist(),**info})
        if term or trunc or info['is_success']:break
    return {'steps':len(rows),'return':total,'initial_distance_m':initial['distance_m'],
        'minimum_distance_m':minimum,'last':info,'samples':rows,
        'motion':approach_motion_summary(start_position,start_joints,env.goal,rows),
        'evaluation_wall_s':time.perf_counter()-started}


def evaluation_clone(env):
    """共享 MuJoCo 模型，评估使用独立状态和执行器。"""
    import mujoco
    from mocap_pose_controller import MocapPoseController

    other = copy.copy(env)
    other.scene = copy.copy(env.scene)
    other.data = other.scene.data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, other.data)
    other.pose_controller = MocapPoseController(other.scene, env.pose_controller.settings)
    return other
