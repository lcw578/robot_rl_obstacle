"""虚拟TCP命令更新方式；不修改机械臂实际qpos/qvel。"""
import numpy as np


def target_update_mode(settings):
    mode=settings.get('target_update','rate_limited')
    if mode not in ('rate_limited','step_interpolated'):
        raise ValueError('pose_controller.target_update须为rate_limited或step_interpolated')
    return mode


def interpolated_step_target(p0,q0,target_p,target_q,index,count):
    """p(k)=p0+(k/N)(p*-p0)，q(k)=SLERP(q0,q*,k/N)。

    第N子步前虚拟目标到达完整请求处；实际TCP由mj_step决定，不承诺到达。
    此命令插补不使用固定线速度/角速度上限，也不直接写机械臂关节状态。
    """
    if type(count) is not int or count<1 or type(index) is not int or not 1<=index<=count:
        raise ValueError('插补子步须满足1≤index≤count')
    p0=np.asarray(p0,float);p1=np.asarray(target_p,float)
    q0=np.asarray(q0,float);q1=np.asarray(target_q,float)
    if p0.shape!=(3,) or p1.shape!=(3,) or q0.shape!=(4,) or q1.shape!=(4,):
        raise ValueError('目标插补须为3维位置和wxyz四元数')
    if not all(np.isfinite(value).all() for value in (p0,p1,q0,q1)):
        raise ValueError('插补输入须有限')
    if min(np.linalg.norm(q0),np.linalg.norm(q1))<1e-12:
        raise ValueError('插补四元数不能为零')
    q0=q0/np.linalg.norm(q0);q1=q1/np.linalg.norm(q1)
    dot=float(q0@q1)
    if dot<0:q1=-q1;dot=-dot
    u=index/count;dot=np.clip(dot,-1,1)
    if dot>.9995:q=(1-u)*q0+u*q1
    else:
        theta=np.arccos(dot)
        q=(np.sin((1-u)*theta)*q0+np.sin(u*theta)*q1)/np.sin(theta)
    return p0+u*(p1-p0),q/np.linalg.norm(q)
