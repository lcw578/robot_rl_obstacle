"""机器人型号范围与任务限位分层：型号显式选定，任务范围只能在其内收紧。"""
import numpy as np

J16_360_PROFILE='aubo_i5_j1_j6_360_v1'
J16_360_DEG={'shoulder_joint':[-360.,360.],'upperArm_joint':[-175.,175.],
             'foreArm_joint':[-175.,175.],'wrist1_joint':[-175.,175.],
             'wrist2_joint':[-175.,175.],'wrist3_joint':[-360.,360.]}


def reference_limits_for_profile(names,reference_rad,profile=None):
    """用户选定官方J1/J6扩展版本；不是任务YAML任意放宽原模型的后门。

    未选profile保持旧快照参考范围。选定后仅改变六轴range，几何/轴向/零位不改。
    官方型号说明：https://github.com/AuboRobot/aubo_description/blob/main/README.md
    版本27：ID1/ID6 ±360°、ID2-5 ±175°。实际控制器仍应读回核对。
    """
    names=list(names);reference=np.asarray(reference_rad,float)
    if (reference.shape!=(len(names),2) or not np.isfinite(reference).all()
            or np.any(reference[:,0]>=reference[:,1])):raise ValueError('参考关节范围无效')
    if profile is None:return reference.copy()
    if profile!=J16_360_PROFILE:raise ValueError('未知robot_joint_limit_profile：'+str(profile))
    if len(names)!=6 or set(names)!=set(J16_360_DEG):raise ValueError('J1/J6扩展profile仅适用于本项目AUBO-i5六轴')
    return np.deg2rad([J16_360_DEG[name] for name in names])


def check_snapshot_limit_profile(manifest,config):
    """旧快照可按显式新profile重建；新profile快照不能静默当作旧机器人加载。"""
    requested=config.get('robot_joint_limit_profile')
    if requested is not None and requested!=J16_360_PROFILE:raise ValueError('未知robot_joint_limit_profile')
    saved=manifest.get('robot_joint_limit_profile')
    if saved is not None and saved!=requested:
        raise ValueError('机器人快照关节型号profile不匹配；请使用配套配置或原始快照重新构建')


def update_position_actuator_ranges(root,names,effective_rad):
    """型号切换后同步角度servo的ctrlrange；不把电机力矩ctrlrange误当角度。

    MuJoCo hinge位置servo的控制量为rad；compiler.angle只转换joint等XML角度字段。
    当前mocap后端把这些servo的kp/kv置0，但范围仍保持一致供后续控制器核对。
    """
    ranges=dict(zip(names,np.asarray(effective_rad,float)))
    actuators=root.find('actuator')
    if actuators is None:return
    for actuator in actuators:
        if actuator.tag=='position' and actuator.get('joint') in ranges:
            actuator.set('ctrlrange',' '.join(f'{value:.16g}' for value in ranges[actuator.get('joint')]))
            actuator.set('ctrllimited','true')


def apply_task_joint_limits(root,names,reference_rad,limits_deg,home_deg):
    """q_i∈[l_model,u_model]∩[l_task,u_task]；配置禁止扩大模型限位。

    reference_rad来自原模型或快照保存的参考范围，不能把上次任务范围
    当作机械臂原始范围。直接写入XML，使编译模型、IK和mocap检查一致。
    """
    names=list(names);reference=np.asarray(reference_rad,float)
    home=np.asarray(home_deg,float)
    if (reference.shape!=(len(names),2) or not np.isfinite(reference).all()
            or np.any(reference[:,0]>=reference[:,1])):
        raise ValueError('参考关节范围必须为有限且下界小于上界的N×2数组')
    if not isinstance(limits_deg,dict) or set(limits_deg)-set(names):
        raise ValueError('task_joint_limits_deg必须按有效机械臂关节名配置')
    effective=reference.copy()
    for name,values in limits_deg.items():
        bounds=np.asarray(values,float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            raise ValueError('无效任务角度范围：'+name)
        bounds=np.deg2rad(bounds);index=names.index(name)
        if bounds[0]<reference[index,0]-1e-10 or bounds[1]>reference[index,1]+1e-10:
            raise ValueError('任务限位不能扩大参考模型范围：'+name)
        effective[index]=bounds
    if home.shape!=(len(names),) or not np.isfinite(home).all():
        raise ValueError('HOME须为对应各关节的有限角度')
    q=np.deg2rad(home)
    if np.any(q<effective[:,0]-1e-10) or np.any(q>effective[:,1]+1e-10):
        raise ValueError('HOME超出任务/模型关节范围')
    compiler=root.find('compiler')
    angle='degree' if compiler is None else compiler.get('angle','degree')
    if angle not in ('degree','radian'):raise ValueError('未知XML角度单位')
    # 先验证全部参数，最后再写XML，避免配置错误时留下半更新状态。
    joints=[root.find(f".//joint[@name='{name}']") for name in names]
    if any(joint is None for joint in joints):raise ValueError('XML缺少机械臂关节')
    for joint,bounds in zip(joints,effective):
        values=np.rad2deg(bounds) if angle=='degree' else bounds
        joint.set('range',' '.join(f'{v:.16g}' for v in values))
        joint.set('limited','true')
    return effective
