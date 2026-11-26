"""独立mocap+weld动力学执行器；不使用IK，不在正常执行期间写机械臂qpos。

目标在基座B系输入，mocap在世界W系更新；mj_step积分出关节状态。
支持历史限速插补与完整周期目标插补；两者、理想重力补偿和安全拒绝
都是本平台适配，不是作者公开的完整执行参数。
"""
import xml.etree.ElementTree as ET
import time
import numpy as np
import mujoco
from env_cluster import _quat_from_mat
from mocap_target_update import target_update_mode,interpolated_step_target

TARGET = 'drl_mocap_target'
WELD = 'drl_mocap_weld'
GEAR_HOLD = 'drl_mocap_gear_hold'


def rotation_error(target, current):
    """计算基座坐标系下的短弧旋转向量 log(R_target R_current^T)。"""
    q = _quat_from_mat(np.asarray(target) @ np.asarray(current).T)
    if q[0] < 0:
        q = -q
    length = np.linalg.norm(q[1:])
    if length < 1e-10:
        return 2 * q[1:]
    return 2 * np.arctan2(length, q[0]) * q[1:] / length


def arm_damping_override(settings):
    """可选的六轴黏性阻尼，单位N·m·s/rad；未配置时保留模型值。

    阻尼力矩 τ_d = -b*qvel，随速度产生制动；不改变关节限位或把速度清零。
    这是仿真参数对照接口，不代表厂家电机已标定。
    """
    if 'arm_joint_damping' not in settings:
        return None
    values = np.asarray(settings['arm_joint_damping'], dtype=float)
    if values.ndim == 0:
        values = np.full(6, float(values))
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('arm_joint_damping须为有限非负标量或六轴数组')
    return values


def strip_mocap_nodes(root):
    """快照重建时仅清除本模块拥有的节点，防止重复添加。"""
    for parent in root.iter():
        for node in list(parent):
            if node.get('name') in (TARGET, WELD, GEAR_HOLD):
                parent.remove(node)


def configure_mocap_xml(root, config, home_position_world, home_rotation_world):
    settings = config['pose_controller']
    mode = target_update_mode(settings)
    damping = arm_damping_override(settings)
    positive_keys = ['physics_dt', 'weld_timeconst_s', 'weld_torquescale_m',
                     'position_tolerance_m', 'rotation_tolerance_rad']
    if mode == 'rate_limited':
        positive_keys.extend(('linear_speed_m_s', 'angular_speed_rad_s'))
    if 'joint_velocity_rad_s' in settings:
        positive_keys.append('joint_velocity_rad_s')
    for key in positive_keys:
        if key not in settings or not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError('无效pose_controller.' + key)
    if settings['weld_timeconst_s'] < 2*settings['physics_dt']:
        raise ValueError('weld时间常数至少是物理积分步长的两倍')
    if settings.get('joint_velocity_rad_s', 0.) > .75 + 1e-10:
        raise ValueError('显式设置的历史作业速度上限须在0.75rad/s内')
    if not np.isfinite(settings['safety_failure_penalty']) or settings['safety_failure_penalty'] > 0:
        raise ValueError('安全失败罚分必须是有限非正数')
    if damping is not None:
        from robot_model import ARM
        arm_joints = [root.find(f".//joint[@name='{name}']") for name in ARM]
        if any(joint is None for joint in arm_joints):
            raise ValueError('设置arm_joint_damping前须有完整六轴关节')
        for joint, value in zip(arm_joints, damping):
            joint.set('damping', str(float(value)))
    strip_mocap_nodes(root)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(settings['physics_dt']))
    option.set('integrator', 'implicitfast')
    # 保留原位置执行器名称/ctrl维度供旧场景工具使用，但去掉机械臂伺服力。
    # 否则它们会把关节拉回HOME，与weld竞争；夹爪servo仍保留。
    for actuator in root.find('actuator'):
        if actuator.get('name') in {f'j{i}' for i in range(1, 7)}:
            if actuator.tag != 'position':
                raise ValueError('当前mocap适配只支持已核对的position机械臂执行器')
            actuator.set('kp', '0'); actuator.set('kv', '0')
            actuator.attrib.pop('dampratio', None)
    if 'gravity_compensation' in settings:
        enabled = settings['gravity_compensation']
        if type(enabled) is not bool:
            raise ValueError('pose_controller.gravity_compensation须为布尔值')
        for body in root.find(".//body[@name='arm_base_mount']").iter('body'):
            body.set('gravcomp', '1' if enabled else '0')
    fmt = lambda x: ' '.join(f'{v:.12g}' for v in x)
    body = ET.SubElement(root.find('worldbody'), 'body', name=TARGET, mocap='true',
        pos=fmt(home_position_world), quat=fmt(_quat_from_mat(home_rotation_world)))
    ET.SubElement(body, 'site', name='drl_mocap_site', type='sphere', size='.004',
                  rgba='1 .1 1 .6', group='4')  # 标记无geom，不参与碰撞。
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    # T_W_tcp ≈ T_W_mocap：位置、方向均跟随，不是只把两个点连接。
    # 显式单位relpose，避免默认采用qpos0时的相对位姿而引入安装偏差。
    ET.SubElement(equality, 'weld', name=WELD, body1=TARGET, body2='tool_tcp',
        relpose='0 0 0 1 0 0 0', solref=f"{settings['weld_timeconst_s']} 1",
        solimp='.95 .99 .001', torquescale=str(settings['weld_torquescale_m']))
    # 接近阶段夹爪固定张开；由物理等式保持，不是每步改qpos。
    ET.SubElement(equality, 'joint', name=GEAR_HOLD, joint1='gear_joint',
        polycoef=f"{config['open_gear_rad']} 0 0 0 0", solref='.01 1')


class MocapPoseController:
    def __init__(self, scene, settings):
        self.scene, self.settings = scene, dict(settings)
        self.target_update=target_update_mode(self.settings)
        self.model, self.data = scene.model, scene.data
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET)
        if self.body < 0:
            raise ValueError('缺少mocap模型节点；必须在模型编译前配置mocap后端')
        self.mocap_id = int(self.model.body_mocapid[self.body])
        self.dofs = self.model.jnt_dofadr[scene.arm_jids]
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.saved = np.zeros(mujoco.mj_stateSize(self.model, self.state_spec))
        from kinematic_guard import KinematicGuard
        self.kinematic_guard = KinematicGuard(scene, settings)
        self.stopped = False
        self.reset()

    def read_state(self):
        p, R = self.scene.tcp_pose()
        return {'position_base_m': p.copy(), 'rotation_base': R.copy(),
                'joints_rad': self.data.qpos[self.scene.arm_qadr].copy(),
                'velocities': self.data.qvel.copy()}

    def reset(self):
        self.stopped = False
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[self.scene.tcp_id]
        self.data.mocap_quat[self.mocap_id] = _quat_from_mat(
            self.data.xmat[self.scene.tcp_id].reshape(3, 3))
        mujoco.mj_forward(self.model, self.data)

    def stop(self):
        self.stopped = True

    def execute_pose(self, target_position, target_rotation, duration_s, collision_substep_rad=None):
        wall_started=time.perf_counter();physics_wall=0.;collision_wall=0.
        from paper_method import slerp
        ptarget, Rtarget = np.asarray(target_position, float), np.asarray(target_rotation, float)
        if (ptarget.shape != (3,) or Rtarget.shape != (3, 3)
                or not np.isfinite(ptarget).all() or not np.isfinite(Rtarget).all()
                or not np.allclose(Rtarget.T@Rtarget, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(Rtarget), 1., atol=1e-6)):
            raise ValueError('mocap目标必须为有限位置与有效旋转矩阵')
        dt = float(self.model.opt.timestep)
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('mocap决策时长须有限且为正')
        count = int(round(duration_s/dt))
        if count < 1 or not np.isclose(count*dt, duration_s, atol=1e-10, rtol=0):
            raise ValueError('control_dt必须为physics_dt的整数倍')
        initial = self.read_state(); self.stopped = False
        mujoco.mj_getState(self.model, self.data, self.saved, self.state_spec)
        p0, R0 = initial['position_base_m'], initial['rotation_base']
        requested_angle = float(np.linalg.norm(rotation_error(Rtarget, R0)))
        initial_q=_quat_from_mat(R0)
        command_p, command_q = p0.copy(), initial_q.copy()
        target_q = _quat_from_mat(Rtarget)
        path, times, velocities = [initial['joints_rad'].tolist()], [0.], [self.data.qvel.tolist()]
        peak_speed, budget, hits, failure = 0., 0., [], None
        peak_joint_speeds=np.zeros(len(self.dofs))
        velocity_violations=[]
        speed_limit = self.settings.get('joint_velocity_rad_s')
        limit_violations = []
        bounds = self.model.jnt_range[self.scene.arm_jids]
        previous = initial['joints_rad'].copy()
        last_tcp = self.data.xpos[self.scene.tcp_id].copy()
        tcp_path_length = 0.
        warnings = self.data.warning.number.copy()
        record_every = max(1, int(round(.02/dt)))
        completed = 0
        self.reset()  # 只对齐虚拟目标，保留实际qpos/qvel，避免上次目标残差成为隐藏控制状态。
        self.kinematic_guard.begin()
        for index in range(count):
            if self.stopped:
                failure = 'stopped'; break
            if self.target_update=='step_interpolated':
                command_p,command_q=interpolated_step_target(p0,initial_q,ptarget,target_q,index+1,count)
            else:
                # 历史默认保持不变；绝不截断实际qvel或瞬移机械臂。
                dp = ptarget-command_p
                command_p += dp*min(1., self.settings['linear_speed_m_s']*dt/max(np.linalg.norm(dp), 1e-12))
                angle = float(np.linalg.norm(rotation_error(Rtarget, _quat_to_mat(command_q))))
                command_q = slerp(command_q, target_q,
                    min(1., self.settings['angular_speed_rad_s']*dt/max(angle, 1e-12)))
            self.data.mocap_pos[self.mocap_id] = self.scene.base_position+self.scene.base_rotation@command_p
            self.data.mocap_quat[self.mocap_id] = _quat_from_mat(self.scene.base_rotation@_quat_to_mat(command_q))
            physics_started=time.perf_counter()
            mujoco.mj_step(self.model, self.data)
            completed += 1
            # mj_step积分后更新FK；禁止通过scene.set_state把速度清零。
            mujoco.mj_kinematics(self.model, self.data)
            physics_wall+=time.perf_counter()-physics_started
            q = self.data.qpos[self.scene.arm_qadr].copy()
            speed = float(np.max(abs(self.data.qvel[self.dofs])))
            peak_joint_speeds=np.maximum(peak_joint_speeds,abs(self.data.qvel[self.dofs]))
            peak_speed = max(peak_speed, speed)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or np.any(self.data.warning.number > warnings)):
                failure = 'physics_warning_or_nonfinite'; break
            collision_started = time.perf_counter()
            hits = self.scene.checker.contacts(self.data, include_mujoco=True)
            collision_wall += time.perf_counter() - collision_started
            if hits:
                break
            if np.any(q < bounds[:, 0]-1e-6) or np.any(q > bounds[:, 1]+1e-6):
                # 回退之前保存越限子步；最终安全姿态不能用来倒推哪个轴曾经越限。
                for j in np.flatnonzero((q < bounds[:, 0]-1e-6) | (q > bounds[:, 1]+1e-6)):
                    limit_violations.append({
                        'joint_number':int(j)+1,
                        'joint_name':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,int(self.scene.arm_jids[j])),
                        'attempted_angle_deg':float(np.degrees(q[j])),
                        'allowed_range_deg':np.degrees(bounds[j]).tolist(),
                        'physics_substep':index+1})
                failure = 'joint_limit'; break
            if speed_limit is not None and speed > speed_limit + 1e-6:
                for j in np.flatnonzero(abs(self.data.qvel[self.dofs]) > speed_limit + 1e-6):
                    velocity_violations.append({
                        'joint_number':int(j)+1,
                        'joint_name':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,int(self.scene.arm_jids[j])),
                        'velocity_rad_s':float(self.data.qvel[self.dofs[j]]),
                        'limit_rad_s':float(speed_limit),
                        'physics_substep':index+1,'time_s':float((index+1)*dt)})
                failure = 'joint_velocity'; break
            if collision_substep_rad is not None and np.max(abs(q-previous)) > collision_substep_rad+1e-9:
                failure = 'collision_resolution'; break
            if self.kinematic_guard.check():
                failure = 'near_singularity'; break
            current_tcp = self.data.xpos[self.scene.tcp_id]
            tcp_path_length += float(np.linalg.norm(current_tcp - last_tcp))
            last_tcp[:] = current_tcp
            budget += float(np.max(abs(q-previous))); previous = q
            if (index+1) % record_every == 0 or index == count-1:
                path.append(q.tolist()); times.append((index+1)*dt); velocities.append(self.data.qvel.tolist())
        collision = bool(hits)
        if collision or failure:
            # 安全拒绝是仿真事务回退，不代表真机可倒退撤销一次运动。
            mujoco.mj_setState(self.model, self.data, self.saved, self.state_spec)
            mujoco.mj_forward(self.model, self.data)
            path, times = [initial['joints_rad'].tolist()]*2, [0., duration_s]
            velocities = [initial['velocities'].tolist()]*2
            budget = 0.
            tcp_path_length = 0.
            self.data.time += duration_s
        final = self.read_state()
        pe = float(np.linalg.norm(ptarget-final['position_base_m']))
        re = float(np.linalg.norm(rotation_error(Rtarget, final['rotation_base'])))
        reached = (not collision and failure is None and pe <= self.settings['position_tolerance_m']
                   and re <= self.settings['rotation_tolerance_rad'])
        return {'execution': 'mocap_weld_mj_step', 'dynamics_simulated': True,
            'execution_wall_s':time.perf_counter()-wall_started,
            'physics_step_wall_s':physics_wall,'collision_query_wall_s':collision_wall,
            'dynamics_validated': False, 'acceleration_limited': False,
            'collision': collision, 'contacts': hits, 'joint_limit': failure=='joint_limit',
            'execution_failure': failure is not None, 'execution_failure_reason': failure,
            'joint_limit_violations':limit_violations,
            'safety_penalty': float(self.settings['safety_failure_penalty']) if failure else 0.,
            'target_position_base_m': ptarget.tolist(), 'target_rotation_base': Rtarget.tolist(),
            # 命令由所选目标更新规则生成；拒绝时保留尝试命令，实际反馈为回退状态。
            'commanded_position_base_m': command_p.tolist(),
            'commanded_rotation_base': _quat_to_mat(command_q).tolist(),
            'commanded_translation_base_m': (command_p-p0).tolist(),
            'commanded_rotation_rad': float(np.linalg.norm(rotation_error(_quat_to_mat(command_q), R0))),
            'kinematic_guard': self.kinematic_guard.report(),
            'actual_position_base_m': final['position_base_m'].tolist(),
            'requested_translation_base_m': (ptarget-p0).tolist(),
            'actual_translation_base_m': (final['position_base_m']-p0).tolist(),
            'tcp_substep_path_length_m': tcp_path_length,
            'requested_rotation_rad': requested_angle,
            'pose_tracking_distance_m': pe, 'pose_tracking_rotation_rad': re,
            'tracking_reached': bool(reached),
            'tracking_status': 'collision' if collision else failure or ('reached' if reached else 'time_budget_exhausted'),
            'execution_duration_s': float(duration_s), 'physics_steps': completed,
            'mocap_target_update':self.target_update,
            'peak_joint_velocity_rad_s': peak_speed, 'joint_path_budget_rad': budget,
            'peak_joint_velocity_by_joint_rad_s':peak_joint_speeds.tolist(),
            'joint_velocity_violations':velocity_violations,
            'executed_joint_waypoints_rad': path, 'executed_waypoint_times_s': times,
            'executed_qvel': velocities}


def _quat_to_mat(q):
    from env_cluster import _quat_to_mat as convert
    return convert(q)
