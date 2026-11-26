"""论文位姿动作、HAF 与 MuJoCo mocap 环境。"""
import numpy as np
from gymnasium import spaces
from gymnasium import Env
from whole_robot_scene import WholeRobotScene
from mocap_pose_controller import MocapPoseController
import mujoco
from env_cluster import _quat_from_mat, _quat_to_mat
import paper_reward

OBS_SIZE=43
POS=slice(0,3);QUAT=slice(3,7);TIME=41
PAPER_DISTANCE = 13
PAPER_NORMAL = slice(14, 17)
PAPER_OBS_SIZE = 19
HAF_REFERENCE = 'haf_reference_position'
ACTION_CONTRACT = 'fused_fixed_reference_v2'


class HomeCollisionError(ValueError):
    """固定起始姿态与场景几何有碰撞。"""


def fusion_settings_from_config(config):
    """训练、诊断、回放使用同一份融合参数；omega仍以整数决策步计。

    α=exp(-omega*t)；物理时间的衰减常数τ=control_dt/omega。
    改omega是独立适配实验，不悄悄把论文的t改成t/T。
    alpha是四元数SLERP权重；完整目标由mocap/weld尝试执行。
    正omega时半衰期为ln(2)/omega步。2对应约0.35步，独立0.08候选约8.66步；
    后者为已有整数步公式的参数对照，不改变HAF算法或强制全程Z指向Pg。
    """
    method=config['paper_method'];steps=config['max_episode_steps']
    omega=float(method['omega'])
    if not np.isfinite(omega) or omega<0:
        raise ValueError('paper_method.omega须为有限非负数')
    if type(method['haf']) is not bool or type(steps) is not int or steps<1:
        raise ValueError('haf须为布尔值，max_episode_steps须为正整数')
    settings = {'omega': omega, 'haf': method['haf'], 'episode_steps': steps,
                'action_contract': ACTION_CONTRACT}
    parameterization = method.get('quaternion_parameterization', 'identity_offset')
    if parameterization == 'normalized':
        settings['quaternion_parameterization'] = parameterization
    elif parameterization == 'identity_offset':
        scale = float(method['quaternion_scale'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('quaternion_scale须为有限正数')
        settings['quaternion_scale'] = scale
    else:
        raise ValueError('无效的quaternion_parameterization')
    layout = method.get('observation', 'extended')
    if layout == 'paper':
        settings['time_index'] = PAPER_OBS_SIZE - 1
    elif layout != 'extended':
        raise ValueError('paper_method.observation须为paper或extended')
    return settings


def check_policy_fusion_compatibility(model,config):
    """检查点内保存着融合参数；不能只改环境YAML就声称换了策略融合。"""
    expected=fusion_settings_from_config(config)
    if (getattr(model.policy,'fusion_settings',None)!=expected
            or getattr(model.actor,'fusion_settings',None)!=expected):
        raise ValueError('策略内保存的融合/动作预算参数与配置不一致；请使用模型配套config.yaml，'
                         '新契约须独立实验，不能仅改YAML回放旧模型')


def unit(q):
    q=np.asarray(q,float)
    n=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(n<1e-12):raise ValueError('不能归一化零向量')
    return q/n


def multiply(a,b):
    """Hamilton积，wxyz；R(a⊗b)=R(a)R(b)。"""
    w,x,y,z=a;v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def slerp(a,b,t):
    a=unit(a);b=unit(b);dot=float(a@b)
    if dot<0:b=-b;dot=-dot
    dot=np.clip(dot,-1,1)
    if dot>.9995:return unit((1-t)*a+t*b)
    theta=np.arccos(dot)
    return unit((np.sin((1-t)*theta)*a+np.sin(t*theta)*b)/np.sin(theta))


def tool_z_to_goal_angle(position, rotation, goal):
    """实际工具Z与当前位置指向Pg的夹角（rad），只记录，不增加奖励。

    β=acos(z·(Pg-p)/(||z|| ||Pg-p||))；已在Pg时方向无定义，返回None。
    它不同于固定目标Z轴误差：论文的X姿态族没有规定唯一的目标Z轴。
    """
    delta=np.asarray(goal)[:3]-np.asarray(position)
    if np.linalg.norm(delta)<1e-8:
        return None
    return float(np.arccos(np.clip(unit(np.asarray(rotation)[:,2])@unit(delta),-1.,1.)))


def approach_motion_summary(start_position,start_joints,goal,samples):
    """实际决策端点的运动摘要；区分累计运动、净位移与有效接近。

    Y误差改善量=|Pg_y-p0_y|-|Pg_y-pfinal_y|；正值表示最终更接近目标Y。
    路程按策略周期端点采样，不冒充物理子步积分路程或能耗。
    """
    positions=np.asarray([np.asarray(start_position).tolist()]+
        [r['tcp_position_base_m'] for r in samples],float)
    joints=np.asarray([np.asarray(start_joints).tolist()]+[r['arm_rad'] for r in samples],float)
    target=np.asarray(goal,float)[:3]
    if (positions.shape!=(len(samples)+1,3) or joints.shape!=(len(samples)+1,6)
            or target.shape!=(3,) or not np.isfinite(positions).all()
            or not np.isfinite(joints).all() or not np.isfinite(target).all()):
        raise ValueError('运动摘要要求有限的基座系位置和六轴关节记录')
    return {'sampling':'actual_decision_endpoints_not_physics_substep_integral',
        'initial_position_base_m':positions[0].tolist(),
        'final_position_base_m':positions[-1].tolist(),
        'goal_position_base_m':target.tolist(),
        'net_translation_base_m':(positions[-1]-positions[0]).tolist(),
        'remaining_error_base_m':(target-positions[-1]).tolist(),
        'absolute_y_error_reduction_m':float(abs(target[1]-positions[0,1])-abs(target[1]-positions[-1,1])),
        'tcp_path_length_m':float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        'joint_total_travel_deg':np.degrees(abs(np.diff(joints,axis=0)).sum(axis=0)).tolist(),
        'joint_net_change_deg':np.degrees(joints[-1]-joints[0]).tolist()}


def shift_her_goals(goals,infos,offset=1e-5):
    """future位置/X目标；将目标向T外微移，使未来状态位于虚拟T内。

    u=unit(b+nz)，g'_p=p_future-epsilon*u。两法向与u点积为正，因而
    p_future-g'_p=epsilon*u同时满足两个半空间。真实任务的式(16)–(20)不改。
    """
    result=np.array(goals,copy=True)
    if result.ndim!=2 or result.shape[1]!=6:raise ValueError('位姿HER须为N×6目标')
    if not np.isfinite(offset) or offset<=0:raise ValueError('HER内侧扰动须为正数')
    for j,record in enumerate(infos):
        ctx=record['paper_reward_context'];b=unit(ctx['hemisphere_normal']);n=unit(ctx['zone_normal'])
        if np.linalg.norm(b+n)<1e-7:raise ValueError('T的两个半空间退化，无法构造内侧HER目标')
        direction=unit(b+n)
        result[j,:3]-=offset*direction
    return result


def pose_reward(achieved, desired, info):
    """按历史 T、实际关节位移与碰撞事件重算 HER 奖励。"""
    achieved = np.asarray(achieved)
    desired = np.asarray(desired)
    if achieved.shape != desired.shape or achieved.ndim < 1 or achieved.shape[-1] != 6:
        raise ValueError('目标须为位置和工具 X 轴组成的六维向量')
    scalar = achieved.ndim == 1
    records = np.asarray([info] if scalar else info, dtype=object)
    if records.shape != ((1,) if scalar else achieved.shape[:-1]):
        raise ValueError('HER info 批次形状不一致')
    records = records.ravel()
    base = paper_reward.relabeled_pose_rewards(
        achieved.reshape(-1, 6), desired.reshape(-1, 6), records, dtype=np.float64)
    safety = np.asarray([row.get('safety_penalty', 0.) for row in records], dtype=np.float64)
    result = (base + safety).astype(np.float32)
    return result[0] if scalar else result.reshape(achieved.shape[:-1])


def reward_replay_info(info):
    from copy import deepcopy
    required = ('paper_reward_context', 'actual_joint_delta_norm_rad', 'collision')
    compact = {key: info[key] for key in required}
    for key in ('safety_penalty', 'TimeLimit.truncated'):
        if key in info:
            compact[key] = info[key]
    return deepcopy(compact)


class PaperPoseGoalEnv(Env):
    """7 维末端动作驱动 AUBO i5 的 mocap 焊接约束。"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, scene_seed=None, render_mode=None, config=None, *, visuals=True,
                 auto_render=True, robot_snapshot=None, pose_seed=None):
        if render_mode not in (None, 'human', 'rgb_array'):
            raise ValueError('未知 render_mode')
        from whole_robot_scene import load_config
        config = load_config() if config is None else config
        self.render_mode = render_mode
        self.auto_render = auto_render
        self.scene = WholeRobotScene(
            config['scene_seed'] if scene_seed is None else scene_seed,
            config, visuals=visuals, robot_snapshot=robot_snapshot, pose_seed=pose_seed)
        self.cfg = self.scene.config
        self.model, self.data = self.scene.model, self.scene.data
        self.method = self.cfg['paper_method']
        self.fusion_settings = fusion_settings_from_config(self.cfg)
        self.paper_observation = self.method.get('observation') == 'paper'
        geo = self.scene.geometry
        self.goal = np.r_[geo.grasp_point, geo.normal]
        self.haf_reference = geo.grasp_point.copy()
        self.paper_context = {
            'desired_goal': self.goal.tolist(), 'radius_m': float(geo.radius),
            'hemisphere_normal': geo.hemisphere_normal.tolist(),
            'zone_normal': geo.zone_normal.tolist(),
        }
        if not self.paper_observation:
            self.plant_points = np.concatenate([self.scene.tomato.kp[n] for n in ('M1','M2','P0','P3','P4')])
        observation_size = PAPER_OBS_SIZE if self.paper_observation else OBS_SIZE + self.model.nv
        self.action_space = spaces.Box(-1., 1., shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            HAF_REFERENCE: spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })
        if not np.isfinite(self.method['translation_step_m']) or self.method['translation_step_m'] <= 0:
            raise ValueError('translation_step_m 须为有限正数')
        self.pose_controller = MocapPoseController(self.scene, self.cfg['pose_controller'])
        self.steps = 0
        self.done = True
        self.last_collision = False
        self.reward_batch_calls = 0
        self.viewer = None
        self.renderer = None
        self._validate_config()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.scene.start_q
        self.scene.set_state()
        self._validate_config()
        self.steps = 0
        self.done = False
        self.last_collision = False
        obs = self._obs()
        if self.render_mode == 'human' and self.auto_render:
            self.render()
        return obs, self._info(obs, False, False, [], 0.)

    def _validate_config(self):
        positive_values = ['control_dt', 'position_tolerance_m', 'axis_tolerance_rad']
        if 'collision_substep_rad' in self.cfg:
            positive_values.append('collision_substep_rad')
        for name in positive_values:
            if not np.isfinite(self.cfg[name]) or self.cfg[name]<=0:
                raise ValueError('无效参数:'+name)
        if type(self.cfg['max_episode_steps']) is not int or self.cfg['max_episode_steps']<1:
            raise ValueError('max_episode_steps必须为正整数')
        q=self.data.qpos[self.model.jnt_qposadr];bounds=self.model.jnt_range
        if not np.isfinite(q).all() or np.any(q<bounds[:,0]) or np.any(q>bounds[:,1]):
            raise ValueError('起点非有限或超过参考模型关节限位')
        hits = self.scene.checker.contacts(self.data, include_mujoco=True)
        if hits:
            raise HomeCollisionError('起点存在碰撞，不能用该配置训练:'+str(hits[:3]))

    def _obs(self):
        p,R=self.scene.tcp_pose();geo=self.scene.geometry
        joints = self.data.qpos[self.scene.arm_qadr]
        time = self.steps / self.cfg['max_episode_steps']
        if self.paper_observation:
            state = np.r_[p, _quat_from_mat(R), joints, np.linalg.norm(p - self.goal[:3]),
                          self.goal[3:], float(self.last_collision), time]
        else:
            state = np.r_[p, _quat_from_mat(R), joints / np.pi, self.data.qpos[self.scene.gripper_qadr],
                          self.plant_points, geo.normal, geo.hemisphere_normal, geo.zone_normal,
                          geo.radius, time, float(self.last_collision), self.data.qvel]
        return {'observation': state.astype(np.float32), 'achieved_goal': np.r_[p, R[:, 0]],
                'desired_goal': self.goal.copy(), HAF_REFERENCE: self.haf_reference.copy()}

    def compute_reward(self,achieved_goal,desired_goal,info):
        if np.asarray(achieved_goal).ndim>1:self.reward_batch_calls+=1
        return pose_reward(achieved_goal,desired_goal,info)

    def _info(self,obs,collision,limit,hits,cost,actual_change=0.):
        p,R=self.scene.tcp_pose()
        record={
            'tcp_position_base_m':p.tolist(),
            # 旧字段指固定目标Z的误差；X姿态族无唯一目标Z，所以保持兼容占位0。
            # 不可用该占位值证明Z朝向Pg，实际朝向误差看新字段。
            'tool_z_error_rad':0.,'tool_z_goal_is_fixed':False,
            'tool_z_to_goal_angle_rad':tool_z_to_goal_angle(p,R,self.goal),
            'collision':bool(collision),
            'joint_limit':bool(limit),'contacts':hits,'action_cost':float(cost),
            'actual_joint_delta_norm_rad':float(actual_change),'scene_seed':self.scene.scene_seed,
            'pose_seed': self.scene.pose_seed,
            'episode_step':int(self.steps),
            'execution':'mocap_weld_mj_step',
            'reward_mode':'paper',
            'tool_x_base':R[:,0].tolist(),'paper_reward_context':self.paper_context.copy(),
            'tcp_calibrated':False,'goal_orientation':'x','goal_pregrasp_angle_deg':None}
        detail = paper_reward.terms(obs['achieved_goal'], obs['desired_goal'], record)
        distance, angle = detail['distance_m'], detail['x_angle_rad']
        record.update(paper_reward_terms=detail, distance_m=distance, axis_error_rad=angle,
                      is_success=(distance < self.cfg['position_tolerance_m']
                                  and angle < self.cfg['axis_tolerance_rad']
                                  and not collision and not limit))
        return record

    def step(self,action):
        if self.done:raise RuntimeError('回合结束后请reset')
        action=np.asarray(action,float)
        if action.shape!=(7,) or not np.isfinite(action).all() or np.any(abs(action)>1.00001):
            raise ValueError('须为7维有限动作，范围[-1,1]')
        p, R = self.scene.tcp_pose()
        # action已完成HAF：Δp_B=translation_step_m*a[:3]，p_req=p_actual+Δp_B。
        # 这是相对实际TCP的增量，用基座轴表达；不是绝对坐标，也不自动朝Pg。
        target = p + self.method['translation_step_m'] * action[:3]
        # q_req=Δq_B⊗q_actual（固定基座轴左乘），不是工具局部轴右乘或四元数相加。
        target_R = _quat_to_mat(multiply(unit(action[3:]), _quat_from_mat(R)))
        q0 = self.data.qpos[self.scene.arm_qadr].copy()
        result = self.pose_controller.execute_pose(target, target_R,
            self.cfg['control_dt'], self.cfg.get('collision_substep_rad'))
        self.steps += 1
        self.data.time = self.steps*self.cfg['control_dt']
        self.last_collision = result['collision']
        actual = float(np.linalg.norm(self.data.qpos[self.scene.arm_qadr]-q0))
        obs = self._obs()
        info = self._info(obs, result['collision'], result['joint_limit'],
                          result['contacts'], actual, actual)
        info.update(result)
        # 只记录进度，不进入奖励或动作；区分策略本就退远与执行未跟随。
        goal_position=self.goal[:3].astype(float)
        initial_distance=float(np.linalg.norm(goal_position-p))
        info['requested_goal_progress_m']=initial_distance-float(np.linalg.norm(goal_position-target))
        actual_position=self.scene.tcp_pose()[0]
        info['actual_goal_progress_m']=initial_distance-float(np.linalg.norm(goal_position-actual_position))
        info['requested_translation_norm_m']=float(np.linalg.norm(target-p))
        if result.get('execution_failure'):
            info['is_success']=False
        info['safety_penalty']=float(result.get('safety_penalty',0.))
        reward = float(np.float32(info['paper_reward_terms']['total'] + info['safety_penalty']))
        horizon = self.steps >= self.cfg['max_episode_steps']
        terminated = bool(result['collision'] or result.get('execution_failure') or horizon)
        info['episode_end_reason'] = ('collision' if result['collision'] else
            result['execution_failure_reason'] if result.get('execution_failure') else
            'finite_horizon' if horizon else None)
        self.done = terminated
        if self.render_mode == 'human' and self.auto_render:
            self.render()
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode=='human':
            if self.viewer is None:
                from mujoco import viewer
                self.viewer=viewer.launch_passive(self.model,self.data)
                self.viewer.cam.lookat[:]=self.data.xpos[self.scene.tcp_id]
                self.viewer.cam.distance=1.6;self.viewer.cam.azimuth=135;self.viewer.cam.elevation=-20
                self.viewer.opt.geomgroup[3]=0
            self.viewer.opt.sitegroup[5] = 1
            if self.viewer.is_running():self.viewer.sync()
        elif self.render_mode=='rgb_array':
            if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=720,width=1000)
            cam=mujoco.MjvCamera();cam.lookat[:]=self.data.xpos[self.scene.tcp_id];cam.distance=1.7;cam.azimuth=135;cam.elevation=-20
            opt=mujoco.MjvOption();opt.geomgroup[3]=0
            opt.sitegroup[5] = 1
            self.renderer.update_scene(self.data,camera=cam,scene_option=opt)
            return self.renderer.render().copy()

    def close(self):
        if self.viewer is not None:self.viewer.close();self.viewer=None
        if self.renderer is not None:self.renderer.close();self.renderer=None
