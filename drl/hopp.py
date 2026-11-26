"""论文式(1)–(5)的HOPP几何；这里只计算目标，不规划机械臂路径。

所有点必须在同一坐标系、使用米。默认半球朝基座-X，沿用论文的放置约定。
HOPP与主茎/果梗首段形成的切割平面垂直；该切割平面不一定等于
场景生成两段果梗的竖直平面，因为主茎现在允许偏离果梗生成平面。
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from env_cluster import _quat_from_mat


class DegenerateHarvestSideError(ValueError):
    """式 (4) 的切割平面无法稳定选侧。"""


def _unit(vector, name):
    vector = np.asarray(vector, dtype=float)
    if (vector.shape != (3,) or not np.isfinite(vector).all()
            or np.linalg.norm(vector) < 1e-8):
        raise ValueError(f"{name}必须是非零有限三维向量")
    return vector / np.linalg.norm(vector)


@dataclass
class HoppGeometry:
    grasp_point: np.ndarray       # Pg：位于首段果梗上的抓取点
    normal: np.ndarray            # nt：HOPP法向，目标工具+X方向
    stem: np.ndarray              # 主茎向上单位方向
    cutting_normal: np.ndarray    # 主茎与果梗首段所定的切割平面法向，不一定是果梗生成平面
    zone_normal: np.ndarray       # 式(4)选侧后的切割平面法向
    hemisphere_normal: np.ndarray # 默认[-1,0,0]选参考系−X侧；并非任意目标位置都等于朝向基座
    radius: float
    eta: float                    # 弧度
    stem_origin: np.ndarray | None = None  # P0，供轴线距离复核与橙色尺寸线显示。

    def describe(self, stem_radius, pre=None):
        """目标几何的数值说明；不使用实际TCP的当前位置冒充目标平面距离。"""
        result = {"grasp_point_m": self.grasp_point.tolist(), "normal": self.normal.tolist(),
                  "eta_deg": float(np.degrees(self.eta)), "zone_radius_m": self.radius}
        if self.stem_origin is not None:
            delta = self.grasp_point - self.stem_origin
            distance = float(np.linalg.norm(np.cross(delta, self.stem)))
            result.update(measured_axis_distance_m=distance,
                          point_to_stem_surface_m=distance - stem_radius,
                          plane_axis_distance_m=float(abs(self.normal @ delta)),
                          plane_stem_parallel_error=float(abs(self.normal @ self.stem)))
        if pre is not None:
            pre = np.asarray(pre, float)
            result.update(pregrasp_point_m=pre.tolist(), standoff_m=float(np.linalg.norm(pre - self.grasp_point)),
                          approach_direction=_unit(self.grasp_point - pre, "预接近方向").tolist(),
                          plane_angle_deg=float(np.degrees(np.arctan2((pre-self.grasp_point) @ self.cutting_normal,
                                                                      (pre-self.grasp_point) @ self.stem)) % 360),
                          pregrasp_plane_error_m=float(abs(self.normal @ (pre - self.grasp_point))))
        return result

    def in_zone(self, point, *, radial_tolerance_m=0.0):
        """严格保留论文式(4)的开边界；Pg本身不属于T。

        后续成功判定应单独定义位置/姿态容差，不能要求精确Pg同时严格属于T。
        """
        # 规划圆可恰好位于球面；仅规划调用允许皮米级径向舍入容差。
        # 默认仍为0，训练/奖励的T判定保持原规则，两个选侧条件均不放宽。
        if not np.isfinite(radial_tolerance_m) or radial_tolerance_m < 0:
            raise ValueError('径向数值容差须为有限非负数')
        delta = np.asarray(point, dtype=float) - self.grasp_point
        # 论文式(3)：Z={x: ||x-Pg||²≤r²，x_x≤Pg_x}。
        # 默认b=(-1,0,0)，所以 b·(x-Pg)≥0 等价于 x_x≤Pg_x。
        # 论文式(4)：T={x∈Z: [sign(n·b)*n]·(x-Pg)>0}。
        # self.zone_normal已在compute_hopp中算好；>0是严格边界，不改成≥0。
        return bool(np.linalg.norm(delta) <= self.radius + radial_tolerance_m
                    and self.hemisphere_normal @ delta >= 0
                    and self.zone_normal @ delta > 0)

    def orientation(self, approach_direction):
        """工具X=nt，Z=HOPP内的接近方向，Y由右手规则确定。

        将输入方向投影到HOPP；这只构造一个候选工具姿态，不保证IK可达。
        """
        direction = _unit(approach_direction, "接近方向")
        # 工程候选姿态（不是论文式(17)奖励函数本身）：
        # X=nt；Z=unit(d-(d·nt)nt)，把接近方向d投影到HOPP；Y=Z×X。
        # R=[X Y Z]按列存三个工具轴在世界/基座系的方向，故右手系X×Y=Z。
        # 式(17)在dg≤φ且pe∈T时：rpos=-[acos((vx·nt)/(||vx|| ||nt||))]²。
        # 因此X与nt同向时该角度惩罚为0；这里直接构造方向，不计算DRL奖励。
        z = _unit(direction - (direction @ self.normal) * self.normal,
                  "接近方向在HOPP内的投影")
        y = np.cross(z, self.normal)
        return np.column_stack((self.normal, y, z))

    def approach_pose(self, current_position, standoff=0.08):
        """从HOPP内选一个T内的预接近点，优先靠近当前TCP。

        候选点在以Pg为中心的圆上，工具Z从预接近点指向Pg。
        此选择是本项目的诊断约定，不是论文的RL策略或避障规划器。
        """
        if not np.isfinite(standoff) or not 0 < standoff <= self.radius:
            raise ValueError("预接近距离必须大于0且不超过采收区域半径")
        candidates = []
        # s=主茎单位方向、n=切割面单位法向，二者正交且都在HOPP方向内。
        # 工程预览圆：Ppre(β)=Pg+L(cosβ*s+sinβ*n)，L=standoff，默认0.08m。
        # 360个角度、8cm距离是工程选择，不是论文公式或训练策略。
        for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            offset = standoff * (np.cos(angle) * self.stem
                                 + np.sin(angle) * self.cutting_normal)
            pre = self.grasp_point + offset
            if self.in_zone(pre, radial_tolerance_m=1e-12):
                # 存(到当前TCP的欧氏距离, 候选点, 朝Pg方向)。只比较几何距离，
                # 不求IK，不检查机械臂能否到达或路径是否碰撞。
                candidates.append((np.linalg.norm(pre - current_position), pre, -offset))
        if not candidates:
            raise ValueError("HOPP内找不到符合式(3)(4)的预接近点")
        # Ppre*=argmin_{Ppre∈候选圆∩T} ||Ppre-Ptcp||；箭头方向=Pg-Ppre。
        _, pre, direction = min(candidates, key=lambda item: item[0])
        return pre, _quat_from_mat(self.orientation(direction))


def angle_interval(values,name):
    """有向角度区间，单位度；不跨越-180/180分界。可用[-180,180]表示整圈。"""
    bounds=np.asarray(values,float)
    if (bounds.shape!=(2,) or not np.isfinite(bounds).all()
            or bounds[0]<-180 or bounds[1]>180 or bounds[0]>bounds[1]):
        raise ValueError(name+'须为[-180,180]内下界不大于上界的两个有限角度')
    return bounds.tolist()


def pregrasp_options(config,standoff=None):
    """整机工程接近扇区；不修改论文HOPP法向或区域T的定义。"""
    options={'standoff_m':.08,'angle_range_deg':[-45.,45.],'angle_step_deg':5.,
             'view_direction_base':[0.,-1.,0.]}
    options.update(config.get('pregrasp',{}))
    if standoff is not None:options['standoff_m']=float(standoff)
    options['angle_range_deg']=angle_interval(options['angle_range_deg'],'pregrasp.angle_range_deg')
    for key in ('standoff_m','angle_step_deg'):
        if not np.isfinite(options[key]) or options[key]<=0:raise ValueError('pregrasp.'+key+'须为有限正数')
    span=options['angle_range_deg'][1]-options['angle_range_deg'][0]
    if np.ceil(span/options['angle_step_deg'])+2>362:
        raise ValueError('接近角采样过密；请增大angle_step_deg，使候选数量不超过约360')
    options['view_direction_base']=_unit(options['view_direction_base'],'正视HOPP的参考方向').tolist()
    return options


def pregrasp_basis(geometry,view_direction=(0.,-1.,0.)):
    """0°=竖直向上在HOPP内的投影；从朝小车的一侧正视该平面，逆时针为正。"""
    normal=geometry.normal
    vertical=np.array([0.,0.,1.])
    # u0=unit(ez-(ez·nt)nt)。倾斜HOPP通常不包含严格世界竖直方向。
    up=_unit(vertical-(vertical@normal)*normal,'HOPP内的向上方向')
    view=_unit(view_direction,'HOPP观察方向')
    # 观察法向nv指向观察者；用nv而不翻转实际工具X=nt。
    # 边缘正对观察者(nv·view=0)时，约定从+nt侧看，保持定义确定。
    view_normal=normal if normal@view>=0 else -normal
    positive=_unit(np.cross(view_normal,up),'HOPP逆时针方向')
    return up,positive,view_normal


def pregrasp_sector_diagnosis(geometry,options,sampled_angles):
    """解析检查圆弧与T的连续交集，区分约束冲突和离散采样遗漏。

    p-Pg=L(cosθ*u0+sinθ*u1)，故每个选侧条件均为A*cosθ+B*sinθ与0比较。
    求出这些正弦函数的零点，将角度区间切开；各小区间内不等式符号恒定。
    这里只诊断，不添加候选、不放宽T，也不改动实际IK/碰撞判定。
    """
    up,positive,_=pregrasp_basis(geometry,options['view_direction_base'])
    sides=[np.array([axis@up,axis@positive],float)
           for axis in (geometry.hemisphere_normal,geometry.zone_normal)]
    def inside(angle):
        t=np.deg2rad(angle);ray=np.array([np.cos(t),np.sin(t)])
        # 1e-12只处理解析根附近的浮点误差；实际候选仍用in_zone的严格规则。
        return bool(sides[0]@ray>=-1e-12 and sides[1]@ray>1e-12)
    def intervals(bounds):
        lower,upper=bounds;cuts=[float(lower),float(upper)]
        for A,B in sides:
            if np.hypot(A,B)<1e-12:continue
            root=np.degrees(np.arctan2(-A,B))
            for k in range(-3,4):
                angle=float(root+180*k)
                if lower<angle<upper:cuts.append(angle)
        cuts=sorted(set(cuts))
        arcs=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a>1e-9 and inside((a+b)/2):arcs.append([a,b])
        isolated=[a for a in cuts if inside(a) and not any(x-1e-9<=a<=y+1e-9 for x,y in arcs)]
        return arcs,isolated
    arcs,isolated=intervals(options['angle_range_deg'])
    full,_=intervals([-180.,180.])
    if options['standoff_m']>geometry.radius:arcs=[];isolated=[];full=[]
    mathematical_samples=[float(a) for a in sampled_angles if inside(a)]
    kind=('continuous_T_empty' if not arcs and not isolated else
          'sampling_missed_T' if not mathematical_samples else 'candidate_margin_or_boundary')
    return {'kind_if_no_candidates':kind,'continuous_T_arcs_deg':arcs,
            'isolated_T_angles_deg':isolated,'full_circle_T_arcs_deg':full,
            'sampled_angles_inside_mathematical_T_deg':mathematical_samples,
            'note':'区间端点是否包含仍以b·offset≥0、zone_normal·offset>0为准；不代表IK或碰撞可行。'}


def pregrasp_failure_message(sector):
    """候选为空的分层提示；不能将采样遗漏误报成连续几何无交集。"""
    diagnosis=sector['continuous_T_diagnosis'];kind=diagnosis['kind_if_no_candidates']
    if kind=='continuous_T_empty':
        reason='指定HOPP接近角扇区与区域T在连续角度上无交集'
    elif kind=='sampling_missed_T':
        reason='HOPP接近角扇区与区域T存在连续交集，但当前角度采样未命中；可核对angle_step_deg'
    else:
        reason='存在T内角度，但候选被边界浮点误差或1微米选侧余量筛除'
    return (reason+'；本次角范围'+str(sector['parameters']['angle_range_deg'])
            +'，整圈T允许弧段约'+str(diagnosis['full_circle_T_arcs_deg'])
            +'度；不自动放宽角度或T')


def pregrasp_candidates(geometry,options):
    """Ppre(theta)=Pg+L*(cos(theta)*u0+sin(theta)*u1)，工具Z指向Pg。

    这是候选姿态的工程限制：theta描述Pg指向预接近点的射线，不是工具Z的射线。
    首先施加角度范围，再与原论文区域T求交；没有候选时明确报告，不放宽T。
    """
    distance=options['standoff_m']
    if not np.isfinite(distance) or not 0<distance<=geometry.radius:
        raise ValueError('预接近距离须在(0,目标区域T半径]内')
    lower,upper=options['angle_range_deg'];step=options['angle_step_deg']
    angles=list(np.arange(lower,upper+1e-10,step))
    if not angles or upper-angles[-1]>1e-8:angles.append(upper)
    if lower<=0<=upper and not any(abs(a)<1e-8 for a in angles):angles.append(0.)
    angles=sorted(angles)
    if upper-lower>=360-1e-8:angles=angles[:-1] # -180与+180是同一射线。
    up,positive,view=pregrasp_basis(geometry,options['view_direction_base'])
    candidates=[];excluded=[]
    for degrees in angles:
        theta=np.deg2rad(degrees);offset=distance*(np.cos(theta)*up+np.sin(theta)*positive)
        pre=geometry.grasp_point+offset
        if not geometry.in_zone(pre, radial_tolerance_m=1e-12) or geometry.zone_normal@offset<1e-6:
            excluded.append(float(degrees));continue
        candidates.append({'angle_deg':float(degrees),'point':pre,'direction':-offset,
                           'rotation':geometry.orientation(-offset)})
    return candidates,{'parameters':options,'zero_up_base':up.tolist(),'positive_base':positive.tolist(),
        'view_normal_base':view.tolist(),'sampled_angles_deg':[float(a) for a in angles],
        'accepted_angles_deg':[c['angle_deg'] for c in candidates],
        'excluded_outside_T_angles_deg':excluded,
        'continuous_T_diagnosis':pregrasp_sector_diagnosis(geometry,options,angles),
        'note':'角度范围与论文区域T取交集；几何候选尚未经过IK或路径验证。'}


def add_pregrasp_sector_markers(root,geometry,options):
    """显示蓝色0°射线及橙色接近扇区边界；site不参与碰撞。"""
    world=root.find('worldbody');up,positive,_=pregrasp_basis(geometry,options['view_direction_base'])
    distance=options['standoff_m'];lower,upper=options['angle_range_deg']
    def line(name,a,b,color):
        if np.linalg.norm(b-a)<1e-10:return
        ET.SubElement(world,'site',name=name,type='capsule',size='.0006',group='5',
            fromto=' '.join(f'{v:.12g}' for v in np.r_[a,b]),rgba=color)
    def point(degrees):
        t=np.deg2rad(degrees)
        return geometry.grasp_point+distance*(np.cos(t)*up+np.sin(t)*positive)
    line('hopp_zero_up',geometry.grasp_point,geometry.grasp_point+1.2*distance*up,'0.1 .45 1 .9')
    for name,degrees in (('lower',lower),('upper',upper)):
        line('hopp_sector_'+name,geometry.grasp_point,point(degrees),'1 .55 .05 .85')
    angles=np.linspace(lower,upper,max(2,int(np.ceil((upper-lower)/5))+1))
    for i in range(len(angles)-1):line('hopp_sector_arc_'+str(i),point(angles[i]),point(angles[i+1]),'1 .55 .05 .65')


def compute_hopp(kp, distance=0.010, radius=0.150, hemisphere_normal=(-1., 0., 0.)):
    """由M1/P0/P3计算Pg、nt与区域T；共线等退化情况明确报错。

    函数默认10mm保留论文参考值；运行入口显式传入tomato_cluster.yaml的
    grasp.d_stem；20mm可用于本项目工具适配，实际值以配置/本次参数快照为准。
    """
    points = [np.asarray(kp[name], dtype=float) for name in ("M1", "P0", "P3")]
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in points):
        raise ValueError("M1/P0/P3必须是有限三维点")
    if not np.isfinite([distance, radius]).all() or min(distance, radius) <= 0:
        raise ValueError("抓取轴线距离和区域半径必须为正数")
    m1, p0, p3 = points
    # 记S=M1-P0、P=P3-P0；s=S/||S||、p=P/||P||。
    # 论文式(1)：η=arccos((S·P)/(||S|| ||P||))=arccos(s·p)。
    stem, peduncle = _unit(m1 - p0, "主茎"), _unit(p3 - p0, "果梗首段")
    # 单位向量恒等式：||p×s||=sinη。η∈[0,π]，这里无需再取绝对值。
    sine = np.linalg.norm(np.cross(stem, peduncle))
    if sine < 1e-6:
        raise ValueError("主茎与果梗近共线，HOPP不稳定")
    # 论文式(2)：Pg=P0+p*(d/sinη)；原式用θ记夹角，此处按式(1)的η解释。
    # 原因：沿果梗走L后，到主茎轴线的垂距为L*sinη=d，故L=d/sinη。
    # d是点到主茎轴线的距离，不是点到主茎表面或整把工具的安全距离。
    along = distance / sine
    if along > np.linalg.norm(p3 - p0):
        raise ValueError(f"抓取点超出果梗首段：轴线偏距{distance * 1000:.1f}mm需沿果梗"
                         f"前进{along * 1000:.1f}mm，首段仅{np.linalg.norm(p3 - p0) * 1000:.1f}mm；需重新选点")
    # 论文式(4)：n=P×S，切割面由P0、P3、M1确定；这里归一化不改变选侧。
    cutting_normal = _unit(np.cross(peduncle, stem), "切割平面法向")
    b = _unit(hemisphere_normal, "半球方向")
    side = float(cutting_normal @ b)
    if abs(side) < 1e-6:
        raise DegenerateHarvestSideError("式(4)选侧退化；请重新采样该模型的位姿")
    # 论文式(5)：nt_raw=S×(P×S)，代码用unit(s×(p×s))保留同一方向。
    # 三重积恒等式：s×(p×s)=p-(p·s)s，即果梗方向在主茎垂直面上的投影。
    # 因此nt⊥s、nt⊥n；HOPP方程为 nt·(x-Pg)=0，到主茎轴线距离为d。
    # nt一般不与果梗p平行；只有η=90°时同向，不应直接用p替代nt。
    return HoppGeometry(
        grasp_point=p0 + peduncle * along,
        normal=_unit(np.cross(stem, np.cross(peduncle, stem)), "HOPP法向"),
        stem=stem, cutting_normal=cutting_normal,
        zone_normal=np.sign(side) * cutting_normal,
        hemisphere_normal=b, radius=float(radius),
        eta=float(np.arccos(np.clip(stem @ peduncle, -1., 1.))), stem_origin=p0.copy())


def add_hopp_markers(root, geometry, pre=None):
    """向模型添加不参与碰撞的HOPP小片、Pg、nt箭头和预接近点。"""
    world = root.find("worldbody")

    def fmt(values):
        return " ".join(f"{x:.12g}" for x in np.asarray(values).ravel())

    # box的局部Z取nt，因此它的XY面就是HOPP。小片仅显示局部平面，不是整个T。
    rotation = np.column_stack((geometry.stem,
                                np.cross(geometry.normal, geometry.stem), geometry.normal))
    ET.SubElement(world, "geom", name="hopp_plane", type="box",
                  pos=fmt(geometry.grasp_point), quat=fmt(_quat_from_mat(rotation)),
                  size="0.12 0.12 0.0003", contype="0", conaffinity="0",
                  group="4", rgba="0 0.8 1 0.18")
    ET.SubElement(world, "site", name="hopp_grasp", type="sphere", size="0.004",
                  pos=fmt(geometry.grasp_point), group="4", rgba="1 0 1 1")
    ET.SubElement(world, "site", name="hopp_normal", type="capsule", size="0.0015",
                  fromto=fmt(np.r_[geometry.grasp_point,
                                   geometry.grasp_point + 0.06 * geometry.normal]),
                  group="4", rgba="1 0 0 1")
    if geometry.stem_origin is not None:
        foot = geometry.stem_origin + geometry.stem * (geometry.stem @ (geometry.grasp_point - geometry.stem_origin))
        ET.SubElement(world, "site", name="hopp_axis_distance", type="capsule", size="0.0006",
                      fromto=fmt(np.r_[foot, geometry.grasp_point]), group="4", rgba="1 0.55 0 1")
        ET.SubElement(world, "site", name="hopp_stem_foot", type="sphere", size="0.0015",
                      pos=fmt(foot), group="4", rgba="1 0.55 0 1")
    if pre is not None and np.linalg.norm(pre - geometry.grasp_point) > 1e-8:
        ET.SubElement(world, "site", name="hopp_pregrasp", type="sphere", size="0.005",
                      pos=fmt(pre), group="4", rgba="0 1 0 1")
        ET.SubElement(world, "site", name="hopp_approach", type="capsule", size="0.0008",
                      fromto=fmt(np.r_[pre, geometry.grasp_point]),
                      group="4", rgba="0 1 0 0.6")
        direction = _unit(geometry.grasp_point - pre, "绿色接近线")
        side = _unit(np.cross(geometry.normal, direction), "绿色箭头侧向")
        head_length = min(.010, .3 * np.linalg.norm(pre - geometry.grasp_point))
        for sign, suffix in ((-1, "left"), (1, "right")):
            tail = geometry.grasp_point - head_length * direction + sign * .4 * head_length * side
            ET.SubElement(world, "site", name=f"hopp_approach_arrow_{suffix}", type="capsule", size="0.0008",
                          fromto=fmt(np.r_[tail, geometry.grasp_point]), group="4", rgba="0 1 0 0.8")
