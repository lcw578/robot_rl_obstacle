"""论文式(16)–(20)的最小奖励；PDF pp.7–8，§2.5.1给出阈值/权重。

长度统一为米，角度为弧度。远距离姿态项按正文取零。
真实任务与HER共用位置、工具X轴和历史T上下文计算奖励。
"""
import numpy as np

PHI_M = .150


def _reward_components(achieved, desired, infos):
    """真实任务和HER共用式(16)–(20)及严格的T边界。"""
    a = np.asarray(achieved, dtype=np.float64)
    g = np.asarray(desired, dtype=np.float64)
    if a.ndim != 2 or a.shape != g.shape or a.shape[1] != 6 or len(infos) != len(a):
        raise ValueError('位姿奖励的输入须为对应的N×6')
    if not np.isfinite(a).all() or not np.isfinite(g).all():
        raise ValueError('位姿目标须有限')

    contexts = [i['paper_reward_context'] for i in infos]
    b = np.asarray([c['hemisphere_normal'] for c in contexts]).reshape(-1, 3)
    zone_normal = np.asarray([c['zone_normal'] for c in contexts]).reshape(-1, 3)
    radii = np.asarray([c['radius_m'] for c in contexts])
    change = np.asarray([i['actual_joint_delta_norm_rad'] for i in infos], dtype=np.float64)
    collision = np.asarray([i.get('collision', False) for i in infos], dtype=bool)
    if not np.isfinite(change).all() or np.any(change < 0):
        raise ValueError('实际关节位移须有限且非负')

    # 使用同一个逐行内积表达式计算单步和批量奖励。
    delta = a[:, :3] - g[:, :3]
    distance = np.linalg.norm(delta, axis=1)
    x_norm = np.linalg.norm(a[:, 3:], axis=1)
    n_norm = np.linalg.norm(g[:, 3:], axis=1)
    if np.any(x_norm < 1e-10) or np.any(n_norm < 1e-10):
        raise ValueError('工具X或目标法向量为零')
    cosine = np.sum(a[:, 3:] * g[:, 3:], axis=1) / (x_norm * n_norm)
    angle = np.arccos(np.clip(cosine, -1, 1))
    inside = ((distance <= radii) & (np.sum(b * delta, axis=1) >= 0)
              & (np.sum(zone_normal * delta, axis=1) > 0))
    near = distance <= PHI_M
    r_goal = np.where(near, -.5 * distance**2, -PHI_M * (distance - .5 * PHI_M))
    r_pos = np.where(near, np.where(inside, -angle**2, -10.), 0.)
    r_ctrl = -change
    r_obs = np.where(collision, -100., 0.)
    total = 10 * r_goal + 2 * r_pos + .2 * r_ctrl + r_obs
    return {'r_goal': r_goal, 'r_pos': r_pos, 'r_ctrl': r_ctrl, 'r_obs': r_obs,
            'total': total, 'distance_m': distance, 'x_angle_rad': angle, 'in_T': inside}


def relabeled_pose_rewards(achieved, desired, infos, *, dtype=np.float32):
    return _reward_components(achieved, desired, infos)['total'].astype(dtype)


def terms(achieved, desired, info, *, translated_goal=False, relabeled_pose=False):
    a = np.asarray(achieved, dtype=np.float64)
    g = np.asarray(desired, dtype=np.float64)
    if a.shape != (6,) or g.shape != (6,):
        raise ValueError('论文奖励要求六维位置/工具X轴目标')
    context = info.get('paper_reward_context')
    if context is None:
        raise ValueError('论文奖励缺少该transition的T区域上下文')
    expected = np.asarray(context['desired_goal'])
    if translated_goal and not relabeled_pose and not np.allclose(g[3:], expected[3:], atol=1e-7, rtol=0):
        raise ValueError('平移HER只重标记目标位置，不能改变物理果梗法向')
    if not (translated_goal or relabeled_pose) and not np.allclose(g, expected, atol=1e-7, rtol=0):
        raise ValueError('直接奖励入口仅接受原场景目标；HER请显式指定重标记')
    values = _reward_components(a[None], g[None], [info])
    return {key: value[0].item() for key, value in values.items()}


def compute_reward(achieved,desired,info):
    a=np.asarray(achieved);g=np.asarray(desired)
    if a.shape!=g.shape or a.ndim<1 or a.shape[-1]!=6:raise ValueError('论文奖励目标形状不一致')
    if a.ndim==1:return np.float32(terms(a,g,info)['total'])
    shape=a.shape[:-1];infos=np.asarray(info,dtype=object)
    if infos.shape!=shape:raise ValueError('批量info与目标的批次形状不一致')
    values=[terms(x,y,i)['total'] for x,y,i in zip(a.reshape(-1,6),g.reshape(-1,6),infos.ravel())]
    return np.asarray(values,dtype=np.float32).reshape(shape)
