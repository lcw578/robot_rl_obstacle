"""AUBO i5 场景资产和关节索引。"""
import mujoco
from project_paths import PROJECT_ROOT

MODEL = PROJECT_ROOT / 'assets/robot_for_tomato_harvest/tomato_picker_standalone.xml'
ARM = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]

def obj(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    if i < 0:
        raise ValueError('缺少对象:'+name)
    return i

def addr(m, name):
    return int(m.jnt_qposadr[obj(m,mujoco.mjtObj.mjOBJ_JOINT,name)])
