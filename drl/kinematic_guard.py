"""可选的雅可比监测/边界保护；不是论文奖励，也不是避障路径规划器。

Jw=[Jp; ell*Jr]，rho=sigma_min(Jw)/sigma_max(Jw)。平移/旋转量纲由
ell(米)统一；阈值只能在同一模型、工作点和ell下比较，不能视为厂家限值。
"""
import numpy as np


def matrix_metrics(jacobian):
    singular = np.linalg.svd(np.asarray(jacobian, float), compute_uv=False)
    largest, smallest = float(singular[0]), float(singular[-1])
    ratio = smallest / max(largest, 1e-15)
    return {'singular_values': singular.tolist(), 'sigma_min': smallest,
            'sigma_max': largest, 'ratio': ratio,
            'condition_number': float(largest / max(smallest, 1e-15))}


def violates_boundary(initial_ratio, candidate_ratio, minimum_ratio):
    """正常起点不得跨界；已在界内的历史姿态允许朝脱离方向运动。

    比较本决策起点；不声称能主动找到脱离路径。1e-10只容纳数值误差。
    """
    return candidate_ratio < min(initial_ratio, minimum_ratio) - 1e-10


class KinematicGuard:
    def __init__(self, scene, controller_settings):
        self.scene = scene
        self.settings = controller_settings.get('kinematic_guard', {})
        self.enabled = self.settings.get('enabled', False)
        self.monitor = self.settings.get('monitor', False) or self.enabled
        self.weight = float(self.settings.get('rotation_weight_m', .15))
        self.minimum = float(self.settings.get('minimum_ratio', .001))
        if (type(self.enabled) is not bool or type(self.monitor) is not bool
                or not np.isfinite([self.weight, self.minimum]).all()
                or self.weight <= 0 or not 0 < self.minimum < 1):
            raise ValueError('无效pose_controller.kinematic_guard配置')
        self.jp = np.zeros((3, scene.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.dofs = scene.model.jnt_dofadr[scene.arm_jids]

    def metrics(self):
        import mujoco
        model, data = self.scene.model, self.scene.data
        # mj_step积分后的cdof可能仍属于积分前；jacBody前明确刷新运动学与惯性坐标。
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacBody(model, data, self.jp, self.jr, self.scene.tcp_id)
        B = self.scene.base_rotation.T
        result = matrix_metrics(np.vstack((B @ self.jp[:, self.dofs],
                                           self.weight * B @ self.jr[:, self.dofs])))
        result['joint3_deg'] = float(np.degrees(data.qpos[self.scene.arm_qadr[2]]))
        return result

    def begin(self):
        self.initial = self.metrics() if self.monitor else None
        self.minimum_observed = self.initial['ratio'] if self.monitor else None
        self.violation = None

    def check(self):
        if not self.monitor:
            return False
        current = self.metrics()
        self.minimum_observed = min(self.minimum_observed, current['ratio'])
        bad = violates_boundary(self.initial['ratio'], current['ratio'], self.minimum)
        if self.enabled and bad:
            self.violation = current
            return True
        return False

    def report(self):
        if not self.monitor:
            return {'monitor': False, 'enabled': False}
        return {'monitor': True, 'enabled': self.enabled, 'rotation_weight_m': self.weight,
                'minimum_ratio': self.minimum, 'initial': self.initial,
                'minimum_observed_ratio': self.minimum_observed,
                'attempted_violation': self.violation, 'final': self.metrics()}
