"""参数化番茄串几何和植物碰撞体。"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np      # 数组和向量计算；下面 np.xxx 都来自这个库
import yaml             # 把 YAML 配置文件读成 Python 字典
from project_paths import CONFIG_DIR


CFG = CONFIG_DIR / 'tomato_cluster.yaml'


# ---------------- 旋转小工具（不引 scipy） ----------------
def _rotmat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """输入旋转轴和旋转角，返回 3×3 旋转矩阵 R（Rodrigues 公式）。

    使用方式：新方向 = R @ 原方向。@ 是矩阵乘法，不是逐项相乘。
    np.ndarray / float 是类型提示，分别表示数组/小数，不执行计算。
    """
    axis = np.asarray(axis, float)
    # 归一化：把轴的长度变为 1，只保留方向。调用者应提供非零轴。
    axis = axis / np.linalg.norm(axis)
    # K 是旋转轴对应的叉乘矩阵；先理解本函数用途，再研究这个公式即可。
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    # np.eye(3) 是 3×3 单位矩阵，相当于“不旋转”。
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotmat_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """按固定轴 X→Y→Z 的顺序旋转；当前场景生成流程没有调用这个工具。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # 作用于列向量时从右往左执行，所以先转的 Rx 写在最右边。
    return Rz @ Ry @ Rx


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    """把同一种旋转从 3×3 矩阵写法转成四元数 (w x y z)。

    MuJoCo 的 quat 属性需要四元数。下面按矩阵数值选择不同等价公式，
    避免除以很小的数；它们不是四种不同的旋转效果。
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    # 单位四元数长度应为 1；归一化可以减少浮点计算误差。
    return q / np.linalg.norm(q)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """单位四元数 (w x y z) → 旋转矩阵；列向量是三个局部轴的方向。"""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _fmt(v) -> str:
    """把数组 [0.1, 0, 0.2] 变成 XML 所需的 '0.100000 0.000000 0.200000'。

    ravel() 把数组展平；:.6f 表示保留 6 位小数；join 用空格连接各数字。
    """
    return " ".join(f"{c:.6f}" for c in np.asarray(v).ravel())


def _vertical_bend_normal(rng):
    """经过P0的竖直果梗平面：法向水平，方位独立随机，不要求包含主茎。

    局部P0为原点，平面方程normal·(x-P0)=0。法向没有Z分量，
    所以平面包含竖直方向。主茎倾斜方向不参与平面方位的选择。
    """
    phi = rng.uniform(0., 2 * np.pi)
    return np.array([np.cos(phi), np.sin(phi), 0.])


class _InfeasiblePeduncleAngle(ValueError):
    """所选平面容不下所选主茎—果梗夹角；允许randomize重采样。"""


def _peduncle_plane_reference(stem_dir, normal, eta):
    """保留eta为真实三维夹角，求平面内的参考方向与有向转角alpha。

    把单位主茎s投影到果梗平面，得到长度r、单位方向t。平面内方向
    p=Rot(normal,alpha)@t满足s·p=r*cos(alpha)=cos(eta)。
    不能直接绕随机normal旋转s，否则s原有的面外分量不会消失。
    """
    if not np.isfinite(eta) or not 0 <= eta <= np.pi:
        raise ValueError("eta_deg是两向量真实夹角，必须在0到180度之间")
    projected = stem_dir - np.dot(stem_dir, normal) * normal
    radius = np.linalg.norm(projected)
    if radius < 1e-10 or abs(np.cos(eta)) > radius + 1e-12:
        raise _InfeasiblePeduncleAngle("所选竖直平面无法满足该主茎—首段果梗夹角")
    return projected / radius, float(np.arccos(np.clip(np.cos(eta) / radius, -1., 1.)))




def _segment_aabb_distance(start, end, half) -> float:
    """线段到原点中心轴对齐盒的距离，连续求最小值，不做离散点采样。

    start/end 是线段的两端点；half 是盒在三个方向的半尺寸。
    例如 half=[0.1,0.02,0.03]，完整盒尺寸就是 [0.2,0.04,0.06]。
    AABB 表示盒的边与坐标轴平行，各轴范围为 [-half,+half]。

    线段 x(t)=start+t*(end-start)，t∈[0,1]。t=0/1 对应起点/终点，
    t=0.5 对应中点；t 是走了线段的多少比例，不是物理时间。
    点到盒的平方距离为
    sum(max(abs(x_i(t))-half_i, 0)**2)。在穿过盒的六个面所在平面时
    分段，每段都是二次函数，只需检查该段的受限极小点。
    """
    start, end, half = (np.asarray(x, dtype=float) for x in (start, end, half))
    direction = end - start
    # breaks 保存分段位置，最少包含线段的起点 0 和终点 1。
    breaks = [0.0, 1.0]
    for axis in range(3):
        # axis=0/1/2 对应 x/y/z；方向为 0 时不会跨越该轴的面。
        if direction[axis] != 0.0:
            for face in (-half[axis], half[axis]):
                # 计算线段与这个盒面所在无限平面的交点参数。
                t = (face - start[axis]) / direction[axis]
                if 0.0 < t < 1.0:
                    breaks.append(float(t))
    breaks = np.unique(breaks)  # 去重并从小到大排序
    min_squared = np.inf       # 先设无穷大，再不断用更小的距离更新
    # zip 将相邻分段点配成 [lo,hi]，覆盖整条线段。
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        midpoint = start + (0.5 * (lo + hi)) * direction
        # active 的三个布尔值表示该段在 x/y/z 哪些方向超出了盒边界。
        active = np.abs(midpoint) > half
        # 在区间内，各轴处于盒内/盒外哪一侧保持不变。
        offset = start[active] - np.clip(midpoint[active], -half[active], half[active])
        # 此处 velocity 是位置相对于参数 t 的变化率，不是机械臂速度。
        velocity = direction[active]
        speed_squared = velocity @ velocity
        # 每段距离平方形如 A*t²+2*B*t+C，极小点为 -B/A。
        # clip 将它限制在 [lo,hi]，同时覆盖极小值落在端点的情况。
        # A=0 时距离不变，取 lo 即可。
        t = (np.clip(-(offset @ velocity) / speed_squared, lo, hi)
             if speed_squared > 0.0 else lo)
        # 每个方向超出盒的部分才产生距离；在盒内的方向记为 0。
        residual = np.maximum(np.abs(start + t * direction) - half, 0.0)
        min_squared = min(min_squared, float(residual @ residual))
    # 之前求的是距离平方，最后开方得到实际距离（米）。
    return float(np.sqrt(min_squared))


def _segment_aabb_interior_length(start, end, half):
    """线段严格进入实心盒内部的长度；停在盒面上的连接点不算吞入。

    用slab区间裁剪解析求交，不靠有限采样，以免漏掉很短的穿入部分。
    """
    start, end, half = (np.asarray(value, float) for value in (start, end, half))
    direction = end - start
    lo, hi = 0., 1.
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if abs(start[axis]) >= half[axis]:
                return 0.
        else:
            enter, leave = sorted(((-half[axis] - start[axis]) / direction[axis],
                                    (half[axis] - start[axis]) / direction[axis]))
            lo, hi = max(lo, enter), min(hi, leave)
    return float(max(0., hi - lo) * np.linalg.norm(direction))


class TomatoClusterScene:
    """管理一串番茄的配置、关键点和 MuJoCo 场景生成过程。

    sc = TomatoClusterScene() 创建对象，自动调用 __init__。
    self 表示“当前这个对象”；self.cfg 等变量保存下来供不同方法共用。
    kp 是字典，例如 kp['P0'] 表示取出关键点 P0 的三维坐标。
    """

    def __init__(self, cfg_path=CFG):
        # 读取配置。例：self.cfg['main_stem']['diameter'] 是主茎直径。
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.kp = {}          # 世界系 7 关键点
        self.kp_local = {}    # 串体局部系（刚体变换后、放置前）
        self.placement = np.zeros(3)
        self._box = {}        # 果穗盒：center 中心、quat 朝向、half 半尺寸（局部系）
        self.generation_info = {}  # 接受前尝试次数、拒绝原因及最终几何指标。

    # ---------------- 1) 关键点随机化 ----------------
    def randomize(self, seed=None) -> dict:
        """拒绝采样：主茎间隙、离地高度和外露果梗/果穗盒关系都通过才接受。

        同 seed 仍确定；旧版误接受的候选会被重采样，故不保证与旧版场景相同。
        """
        # seed 是随机序列的“起点编号”：42→重复同一串；None→重新随机。
        # 随机生成发生在这里，查看器本身不会自动生成新场景。
        rng = np.random.default_rng(seed)
        c = self.cfg  # 简短别名，与 self.cfg 是同一个配置字典
        rejected = {}
        for attempt in range(1, 101):
            # _ 表示循环次数本身不用；最多尝试 100 个候选形状。
            try:
                kp, box = self._sample_local(rng)
            except _InfeasiblePeduncleAngle:
                rejected["plane_eta_incompatible"] = rejected.get("plane_eta_incompatible", 0) + 1
                continue  # 平面与eta不相容，重采样；不偷偷改变eta的定义。
            # uniform(lo,hi) 在范围内取随机数；* 把 [lo,hi] 拆成两个参数。
            # placement 是整串番茄在世界坐标中的平移量，顺序为 x/y/z。
            placement = np.array([
                rng.uniform(*c["randomize"]["place_x"]),
                rng.uniform(*c["randomize"]["place_y"]),
                rng.uniform(*c["randomize"]["place_z"])])
            # 盒沿世界Z的半高度=三个局部半尺寸在Z方向投影绝对值之和。
            # 即使后续启用果穗倾斜，也能检查全部8个顶点中的最低点。
            box_bottom = (box["center"][2]
                          - np.abs(_quat_to_mat(box["quat"])[2]) @ box["half"]
                          + placement[2])
            # 连接端允许有限胶囊圆头接合，但不能让盒子吞掉外露果梗。
            # 这里只检查场景合理性，并没有检查机械臂整条运动路径是否安全。
            peduncle_metrics = self._peduncle_box_metrics(kp, box)
            reasons = []
            if self._clearance_box_stem(kp, box) < .005:
                reasons.append("stem_box_clearance")
            if box_bottom < .05:
                reasons.append("box_ground_clearance")
            if not peduncle_metrics["valid"]:
                reasons.append("peduncle_box_overlap")
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            if not reasons:
                break  # 找到合格候选，提前结束尝试
        else:
            # Python 的 for...else：100 次都没有 break 才进入这里。
            raise RuntimeError(f"重采样100次未通过有效性检查，请检查yaml参数；拒绝原因：{rejected}")
        self._box = box
        self.placement = placement
        self.kp_local = kp
        self.generation_info = {"attempts": attempt, "rejection_counts": rejected,
                                "peduncle_box": peduncle_metrics}
        # 字典推导式：遍历每个关键点 k，把局部坐标 v 加上同一平移量。
        self.kp = {k: v + placement for k, v in kp.items()}
        return self.kp

    def _sample_local(self, rng):
        """局部系采样：P0=原点，主茎方向接近 +z → 返回 (kp, box)。

        - 主茎倾角、果穗下垂倾角按YAML配置。
        - 两段果梗在经过P0、方位独立随机的竖直平面内。
        - 主茎可偏离该平面；eta仍指主茎与果梗首段的真实夹角。
        """
        c = self.cfg

        # 第一步：生成主茎。从竖直方向 [0,0,1] 出发，再稍微旋转。
        # deg2rad 把度转换成弧度；倾角范围统一放在YAML，不再写死在代码里。
        stem_tilt = np.deg2rad(rng.uniform(*c["main_stem"].get("tilt_deg", [-15, 15])))
        # stem_az 是水平旋转轴的方位，不等于主茎倾斜方向本身的方位。
        stem_az = rng.uniform(0, 2 * np.pi)
        stem_dir = _rotmat_axis_angle(
            [np.cos(stem_az), np.sin(stem_az), 0], stem_tilt) @ np.array([0, 0, 1.0])
        # P0 在原点，单位方向 × 长度直接得到端点；负号表示往相反方向走。
        m1 = stem_dir * rng.uniform(*c["main_stem"]["seg_above"])
        m2 = -stem_dir * rng.uniform(*c["main_stem"]["seg_below"])

        # 第二步：独立选择经过P0的竖直果梗平面，不强制主茎在其中。
        plane_normal = _vertical_bend_normal(rng)

        # 第三步：先投影主茎，解出平面内转角，保证首段既在平面内又满足eta。
        eta = np.deg2rad(rng.uniform(*c["peduncle"]["eta_deg"]))
        plane_reference, angle1 = _peduncle_plane_reference(stem_dir, plane_normal, eta)
        # p1/p2 是单位方向向量，不是论文中的 P1/P2 关键点。
        p1 = _rotmat_axis_angle(plane_normal, angle1) @ plane_reference
        p3 = p1 * rng.uniform(*c["peduncle"]["len_first"])

        # 第四步：第二段 P3→P4，在同一弯曲平面里继续转过 droop。
        droop = np.deg2rad(rng.uniform(*c["peduncle"]["droop_deg"]))
        # 270°是从平面内主茎投影方向算起的有向累计转角，不是arccos夹角。
        max_bend = float(c["peduncle"].get("max_bend_deg", 270.0))
        if not np.isfinite(max_bend) or not 0 < max_bend <= 360:
            raise ValueError("max_bend_deg必须大于0且不超过360度")
        if angle1 > np.deg2rad(max_bend):
            raise _InfeasiblePeduncleAngle("首段平面转角已超过累计转角上限")
        angle2 = min(angle1 + droop, np.deg2rad(max_bend))
        p2 = _rotmat_axis_angle(plane_normal, angle2) @ plane_reference
        # 第二段从 P3 出发，所以要“P3 + 第二段位移”，而不是从原点出发。
        p4 = p3 + p2 * rng.uniform(*c["peduncle"]["len_second"])

        # 第五步：一个盒子代表整串果实，这里不逐颗创建番茄球体。
        # a 是沿下垂方向的完整长度，s1/s2 是另外两个方向的完整宽度。
        a = rng.uniform(*c["cluster"]["size_along"])
        s1 = rng.uniform(*c["cluster"]["size_side1"])
        s2 = rng.uniform(*c["cluster"]["size_side2"])
        # 按配置取果穗倾角；若设为[0,0]才始终竖直向下。
        tilt = np.deg2rad(rng.uniform(*c["cluster"]["hang_tilt_deg"]))

        # 果穗主轴：从竖直向下开始
        v = np.array([0, 0, -1.0])
        # 倾斜轴 = 垂直于（竖直向下， 果梗方向）
        down = np.array([0, 0, -1.0])
        tilt_axis = np.cross(down, p2)
        if np.linalg.norm(tilt_axis) > 1e-6:
            tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)
            v = _rotmat_axis_angle(tilt_axis, tilt) @ v
        # 否则 p2 也竖直，保持 v 不变

        # 用 v、e1、e2 建立盒子的三个互相垂直的单位轴。
        # ref 是辅助方向，要避开与 v 几乎平行的情况，否则叉积会接近零。
        ref = np.array([1, 0, 0.0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0.0])
        e1 = np.cross(v, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        # 盒上端中心接在 P4，再沿下垂方向走半个长度，到达盒中心。
        center = p4 + v * (a / 2)
        # 此处 half 是“中心到某顶点”的位移向量；不同于下面 box['half']。
        half = v * a / 2 + e1 * s1 / 2 + e2 * s2 / 2
        # 保存七个关键点；center±half 得到一对相对顶点 TA/TB。
        kp = {"M1": m1, "M2": m2, "P0": np.zeros(3), "P3": p3, "P4": p4,
              "TA": center - half, "TB": center + half}
        # box['half'] 是盒沿自身三轴的半尺寸，MuJoCo box 的 size 要用它。
        # column_stack 将三个轴作为列组成旋转矩阵，再换成四元数。
        box = {"center": center, "half": np.array([a, s1, s2]) / 2,
               "quat": _quat_from_mat(np.column_stack([v, e1, e2]))}
        return kp, box

    def _peduncle_box_metrics(self, kp, box):
        """检查外露果梗不被果穗盒吞入；P4仍接在盒上端中心。

        中心轴线除终点接触盒面外不得进入盒内部。胶囊有半径，因此最后
        attachment_length的一小段允许与连接处接合，其余外露胶囊不可穿盒。
        这是一项生成器几何约束，不是把果梗/果穗列入机械臂允许碰撞表。
        """
        attachment = float(self.cfg["cluster"].get("peduncle_attachment_length", .010))
        if not np.isfinite(attachment) or attachment < 0:
            raise ValueError("peduncle_attachment_length必须是有限非负长度（米）")
        radius = self.cfg["peduncle"]["diameter"] / 2
        rotation = _quat_to_mat(box["quat"])
        a, b, end = [rotation.T @ (kp[key] - box["center"]) for key in ("P0", "P3", "P4")]
        length = np.linalg.norm(end - b)
        if attachment >= length:
            raise ValueError("peduncle_attachment_length必须小于第二段果梗长度，不能豁免整段")
        exposed_end = end - (end - b) * (attachment / length)
        first = _segment_aabb_distance(a, b, box["half"]) - radius
        second = _segment_aabb_distance(b, exposed_end, box["half"]) - radius
        inside1 = _segment_aabb_interior_length(a, b, box["half"])
        inside2 = _segment_aabb_interior_length(b, end, box["half"])
        return {"first_surface_clearance_m": float(first),
                "second_exposed_surface_clearance_m": float(second),
                "first_axis_inside_box_m": inside1, "second_axis_inside_box_m": inside2,
                "attachment_length_m": attachment,
                "valid": bool(min(first, second) >= -1e-8 and max(inside1, inside2) <= 1e-8)}

    def _clearance_box_stem(self, kp, box) -> float:
        """果穗盒到主茎 capsule 的分离裕量（m），扣除主茎半径。

        先在盒局部系求轴线线段到实心盒的连续最短距离，再扣半径。
        正值为表面间隙，0 为相切，负值表示重叠（非通用穿透深度）。
        kp 和 box 必须处于同一坐标系。
        """
        R = _quat_to_mat(box["quat"])
        c, h = box["center"], box["half"]
        # 减中心再反向旋转：把轴线端点变换到盒局部坐标系，盒就与坐标轴对齐。
        # R.T 是旋转矩阵的转置，也就是逆旋转；旋转和平移不会改变距离。
        start = R.T @ (kp["M2"] - c)
        end = R.T @ (kp["M1"] - c)
        axis_distance = _segment_aabb_distance(start, end, h)
        # capsule 是给线段加上厚度，必须减掉半径才是表面间隙。
        # 例如轴线距盒 12mm、主茎半径 7mm，实际间隙为 5mm。
        return axis_distance - self.cfg["main_stem"]["diameter"] / 2

    def plant_body(self, segment_type='capsule'):
        """用局部关键点拼装固定植物，交给整机 MJCF 放置。"""
        body = ET.Element('body', name='bunch', pos=_fmt(self.placement))
        stem_radius = self.cfg['main_stem']['diameter'] / 2
        peduncle_radius = self.cfg['peduncle']['diameter'] / 2
        segments = (
            ('stem', 'M1', 'M2', stem_radius, '0.25 0.5 0.25 1'),
            ('peduncle1', 'P0', 'P3', peduncle_radius, '0.3 0.6 0.3 1'),
            ('peduncle2', 'P3', 'P4', peduncle_radius, '0.3 0.6 0.3 1'),
        )
        for name, start, end, radius, color in segments:
            endpoints = np.r_[self.kp_local[start], self.kp_local[end]]
            ET.SubElement(body, 'geom', name=name, type=segment_type, fromto=_fmt(endpoints),
                          size=str(radius), contype='2', conaffinity='1', rgba=color)
        ET.SubElement(body, 'geom', name='cluster', type='box', pos=_fmt(self._box['center']),
                      quat=_fmt(self._box['quat']), size=_fmt(self._box['half']),
                      contype='2', conaffinity='1', rgba='0.85 0.15 0.15 0.9')
        return body
