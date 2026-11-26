"""AUBO i5、工具碰撞代理与 HOPP 植物场景。"""
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import yaml
from robot_model import MODEL, ARM, obj, addr
from env_cluster import TomatoClusterScene, _quat_to_mat, _quat_from_mat, _segment_aabb_distance
from mesh_geometry import slab_bounds
from hopp import (compute_hopp,add_hopp_markers,angle_interval,pregrasp_options,
                  pregrasp_candidates,add_pregrasp_sector_markers)
from project_paths import CONFIG_DIR, DEFAULT_CONFIG, repo_relative

SCENE_VERSION = 'row_harvest_side_top_approach_v4'
STEM_GUARD = 'stem_axis_guard'
TOOL = {'gripper_base','gear_link','right_finger_link','left_finger_link'}
ADJACENT = {frozenset(p) for p in [('arm_base_mount','shoulder_Link'),
    ('shoulder_Link','upperArm_Link'),('upperArm_Link','foreArm_Link'),
    ('foreArm_Link','wrist1_Link'),('wrist1_Link','wrist2_Link'),('wrist2_Link','wrist3_Link'),
    ('wrist3_Link','gripper_base'),('gripper_base','gear_link'),
    ('gripper_base','right_finger_link'),('gripper_base','left_finger_link'),
    ('gear_link','right_finger_link'),('gear_link','left_finger_link')]}


def fmt(x):
    return ' '.join(f'{v:.12g}' for v in np.asarray(x).ravel())


def load_config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def configure_low_memory_display(scene):
    """让精简快照中的透明机械臂碰撞网格可见；只改显示，不改碰撞几何/关节。

    原XML有独立外观网格，碰撞副本alpha=0；删去外观后不能继续把机械臂画成透明。
    """
    model=scene.model;count=0
    for g in np.flatnonzero((model.geom_contype==1)&(model.geom_rgba[:,3]<.001)):
        same_body=(model.geom_bodyid==model.geom_bodyid[g])
        has_visual=same_body&(model.geom_contype==0)&(model.geom_conaffinity==0)&(model.geom_rgba[:,3]>.001)
        if np.any(has_visual):continue # 完整CAD已有外观时，不能再叠一层导致闪烁。
        body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))
        if body in {'shoulder_Link','upperArm_Link','foreArm_Link','wrist1_Link','wrist2_Link','wrist3_Link'}:
            model.geom_rgba[g]=[.68,.72,.8,1.];model.geom_group[g]=1;count+=1
    return count


def harvest_hemisphere(config):
    """式(3)(4)的半球方向b，在基座B系给出；与工具轴、观察角正负方向分开。

    论文参考坐标使用-X；整机植株在+Y，朝车侧应显式使用-Y。
    不提供配置的历史夹具仍保留论文-X约定，不改动compute_hopp默认值。
    """
    direction=np.asarray(config.get('harvest_region',{}).get('hemisphere_direction_base',[-1.,0.,0.]),float)
    if (direction.shape!=(3,) or not np.isfinite(direction).all()
            or abs(direction[2])>1e-10 or np.linalg.norm(direction)<1e-10):
        raise ValueError('harvest_region.hemisphere_direction_base须为非零水平三维方向')
    return direction/np.linalg.norm(direction)


def plant_constraint_options(config):
    """整机任务约束；历史夹具可显式关闭，正式配置默认启用。"""
    options=dict(config.get('plant_constraints',{}))
    # front_only是历史开关；新的可配置方位扇区可以超过±90°，因此用更准确的名称。
    legacy=options.pop('front_only',None)
    options.setdefault('azimuth_filter_enabled',True if legacy is None else legacy)
    for key in ('azimuth_filter_enabled','infinite_stem'):
        options.setdefault(key,True)
        if type(options[key]) is not bool:raise ValueError(key+'须为true/false')
    options['azimuth_range_deg']=angle_interval(options.get('azimuth_range_deg',[-90.,90.]),'plant_constraints.azimuth_range_deg')
    if options['azimuth_range_deg'][0]==options['azimuth_range_deg'][1]:raise ValueError('番茄方位扇区必须有正宽度')
    options.setdefault('front_reference','row_direction')
    if options['front_reference'] not in ('row_direction','observer'):
        raise ValueError('front_reference须为row_direction或observer')
    options.setdefault('front_direction_base',[0.,-1.,0.])
    front=np.asarray(options['front_direction_base'],float)
    if (front.shape!=(3,) or not np.isfinite(front).all()
            or abs(front[2])>1e-10 or np.linalg.norm(front[:2])<1e-10):
        raise ValueError('front_direction_base须为非零的水平三维方向，例如[0,-1,0]')
    options['front_direction_base']=(front/np.linalg.norm(front)).tolist()
    options.setdefault('observer_xy_base_m',[0.,0.])
    observer=np.asarray(options['observer_xy_base_m'],float)
    if observer.shape!=(2,) or not np.isfinite(observer).all():raise ValueError('observer_xy_base_m须为两个有限数')
    options['observer_xy_base_m']=observer.tolist()
    options.setdefault('max_generation_attempts',200)
    if type(options['max_generation_attempts']) is not int or not 1<=options['max_generation_attempts']<=2000:
        raise ValueError('max_generation_attempts须为1至2000的整数')
    return options


def azimuth_sector_margin(radial_points,bounds):
    """凸形状的整个水平投影是否位于角度扇区内，返回角度余量。

    不能只逐个检查角点：±120°扇区不是凸集，盒子角点都合格时，边仍可能穿过背侧禁区。
    对不包含轴线原点的凸形状，角度占据是连续短弧：按角排序，去掉最大空隙，
    得到覆盖整个形状的角弧；它必须整体落在指定区间。穿轴另由主茎间隙检查拒绝。
    """
    lower,upper=bounds
    if upper-lower>=360-1e-8:return 180.
    points=np.asarray(radial_points,float)
    if np.any(np.linalg.norm(points[:,:2],axis=1)<1e-10):return -180.
    # 输入已经以正面方向作为角度0°，坐标分别为(逆时针分量,正面分量)。
    angles=np.sort(np.degrees(np.arctan2(points[:,0],points[:,1]))%360)
    gaps=np.diff(np.r_[angles,angles[0]+360]);gap=int(np.argmax(gaps))
    start=float(angles[(gap+1)%len(angles)]);span=float(360-gaps[gap])
    if span>=180-1e-8:return -180. # 凸投影包含轴线原点或贴过原点，无法限制在局部扇区。
    return max(min(start+shift-lower,upper-(start+shift+span)) for shift in (-360.,0.,360.))


def plant_side_metrics(tomato,observer_xy=(0.,0.),front_direction=None,azimuth_range=(-90.,90.)):
    """检查朝小车的半侧；方向均在B系，长度为米、方位角为度。

    front_direction给定时，全行采用同一正面方向（本平台为-Y）；否则使用
    P0指向observer_xy的方向。两者在植株行的左右边缘并不相同。
    """
    kp=tomato.kp;p0=kp['P0'];stem=kp['M1']-kp['M2'];stem=stem/np.linalg.norm(stem)
    toward=(np.r_[np.asarray(observer_xy)-p0[:2],0.] if front_direction is None
            else np.asarray(front_direction,float).copy())
    if np.linalg.norm(toward)<1e-10:raise ValueError('车侧参考点不能位于主茎正下方，无法区分前后')
    toward/=np.linalg.norm(toward)
    if abs(stem[2])<1e-8:raise ValueError('当前车侧生成规则要求主茎非水平')
    # 在任一点P的高度，主茎轴点为A=P0+s*(Pz-P0z)/sz。
    # “朝小车”即 f·(P-A)>=0。写成平面：n·(P-P0)>=0，
    # n=unit([fx,fy,-(fx*sx+fy*sy)/sz])；故n·s=0，分界面包含整条倾斜主茎。
    # 主茎竖直且f=-Y时，条件简化为 Py<=P0y。
    normal=toward-np.array([0.,0.,(toward@stem)/stem[2]])
    normal/=np.linalg.norm(normal)
    # 用P3相对“同高度主茎轴点”的水平向量r计算有向方位角：
    # theta=atan2(fx*ry-fy*rx, fx*rx+fy*ry)，从+Z俯视逆时针为正。
    # f=-Y时：theta=atan2(rx,-ry)。theta∈[-90°,90°]正好选出朝车半侧。
    first=kp['P3']-p0;radial=first-stem*(first[2]/stem[2])
    if np.linalg.norm(radial[:2])<1e-10:raise ValueError('第一段果梗沿主茎轴，无法定义前后方位角')
    azimuth=float(np.rad2deg(np.arctan2(toward[0]*radial[1]-toward[1]*radial[0],toward@radial)))
    box=tomato._box;rotation=_quat_to_mat(box['quat']);center=box['center']+tomato.placement
    # OBB八个角点投影的最小值=n·(c-P0)-Σ_i |n·R_i|h_i，不仅检查中心。
    box_front=float(normal@(center-p0)-np.abs(normal@rotation)@box['half'])
    peduncle_front=min(float(normal@(kp[k]-p0)) for k in ('P3','P4'))
    # 相对“同高度轴点”的水平投影是线性映射，可用OBB八角和果梗端点求整个角度占据。
    def sector_points(points):
        relative=np.asarray(points)-p0
        radial=relative-np.outer(relative[:,2]/stem[2],stem)
        lateral=toward[0]*radial[:,1]-toward[1]*radial[:,0]
        return np.column_stack((lateral,radial@toward))
    signs=2*np.asarray(list(np.ndindex(2,2,2)))-1
    corners=center+(signs*box['half'])@rotation.T
    box_sector=azimuth_sector_margin(sector_points(corners),azimuth_range)
    peduncle_sector=azimuth_sector_margin(sector_points([kp['P3'],kp['P4']]),azimuth_range)
    radius=tomato.cfg['main_stem']['diameter']/2
    # 这条线段覆盖整个盒子的轴向投影，故线段到盒距离等于无限轴线到盒距离。
    span=np.linalg.norm(center-p0)+np.linalg.norm(box['half'])+1.
    a=rotation.T@(p0-span*stem-center);b=rotation.T@(p0+span*stem-center)
    box_gap=float(_segment_aabb_distance(a,b,box['half'])-radius)
    # 第二段果梗在主茎垂直面上的投影，检查是否返回并穿入延长的主茎。
    u=kp['P3']-p0;u-=stem*(u@stem)
    v=kp['P4']-p0;v-=stem*(v@stem);delta=v-u
    t=float(np.clip(-(u@delta)/(delta@delta),0,1)) if delta@delta>1e-20 else 0.
    second_gap=float(np.linalg.norm(u+t*delta)-radius-tomato.cfg['peduncle']['diameter']/2)
    return {'normal_toward_vehicle_base':normal.tolist(),'front_direction_base':toward.tolist(),
        'first_peduncle_azimuth_deg':azimuth,'box_min_front_m':box_front,
        'box_sector_margin_deg':box_sector,'peduncle_sector_margin_deg':peduncle_sector,
        'peduncle_min_front_m':peduncle_front,'infinite_stem_box_clearance_m':box_gap,
        'infinite_stem_second_peduncle_clearance_m':second_gap}

def workspace_bounds(workspace):
    if workspace.get('reference') != 'grasp_point' or workspace.get('frame') != 'arm_base':
        raise ValueError('target_workspace必须以arm_base系的grasp_point为参考')
    bounds = np.asarray([workspace[axis] for axis in ('x', 'y', 'z')], float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('目标生成范围须为三组有限且下界小于上界的数值')
    if type(workspace.get('contain_bunch', False)) is not bool:
        raise ValueError('target_workspace.contain_bunch须为true/false')
    return bounds


def bunch_bounds(tomato, grasp_point):
    """基座系中包住Pg、果梗胶囊和果穗盒的轴对齐边界。"""
    box = tomato._box
    center = tomato.placement + box['center']
    extent = np.abs(_quat_to_mat(box['quat'])) @ box['half']
    peduncle = np.asarray([tomato.kp[point] for point in ('P0', 'P3', 'P4')])
    radius = tomato.cfg['peduncle']['diameter'] / 2
    lower = np.minimum.reduce([grasp_point, center - extent, peduncle.min(axis=0) - radius])
    upper = np.maximum.reduce([grasp_point, center + extent, peduncle.max(axis=0) + radius])
    return np.column_stack((lower, upper))


def sample_task_tomato(scene_seed,config):
    """在基座系采样Pg，并按配置为完整果穗盒与果梗留出边界余量。"""
    if config.get('paper_training',{}).get('dataset_manifest'):
        from paper_dataset import load_digital_model
        return load_digital_model(scene_seed,config)
    options=plant_constraint_options(config);hemisphere=harvest_hemisphere(config)
    workspace = config.get('target_workspace')
    bounds = None
    if workspace is not None:
        bounds = workspace_bounds(workspace)
        placement_rng = np.random.default_rng(np.random.SeedSequence(scene_seed, spawn_key=(20261104,)))
        placement_fraction = placement_rng.random(3)
    shape_rng=np.random.default_rng(np.random.SeedSequence(scene_seed,spawn_key=(20261106,)))
    tomato=TomatoClusterScene();rejected={}
    distance = config.get('plant_geometry', {}).get('grasp_axis_distance_m')
    if distance is not None:
        tomato.cfg['grasp']['d_stem'] = distance
    for attempt in range(1,options['max_generation_attempts']+1):
        seed=scene_seed if attempt==1 else int(shape_rng.integers(0,2**63))
        tomato.randomize(seed)
        try:
            geometry=compute_hopp(tomato.kp,tomato.cfg['grasp']['d_stem'],tomato.cfg['grasp']['zone_r'],hemisphere)
        except ValueError:
            rejected['hopp_not_defined']=rejected.get('hopp_not_defined',0)+1;continue
        if bounds is not None:
            feasible = bounds
            if workspace.get('contain_bunch', False):
                relative = bunch_bounds(tomato, geometry.grasp_point) - geometry.grasp_point[:, None]
                feasible = bounds - relative
                if np.any(feasible[:, 0] > feasible[:, 1]):
                    reason = 'bunch_too_large_for_workspace'
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
            target = feasible[:, 0] + placement_fraction * (feasible[:, 1] - feasible[:, 0])
            shift=target-geometry.grasp_point;tomato.placement+=shift
            tomato.kp={key:value+shift for key,value in tomato.kp.items()}
            geometry=compute_hopp(tomato.kp,tomato.cfg['grasp']['d_stem'],tomato.cfg['grasp']['zone_r'],hemisphere)
        front=options['front_direction_base'] if options['front_reference']=='row_direction' else None
        metrics=plant_side_metrics(tomato,options['observer_xy_base_m'],front,options['azimuth_range_deg'])
        reasons=[]
        lower,upper=options['azimuth_range_deg']
        if options['azimuth_filter_enabled'] and (not lower-1e-8<=metrics['first_peduncle_azimuth_deg']<=upper+1e-8
                or min(metrics['box_sector_margin_deg'],metrics['peduncle_sector_margin_deg'])<-1e-8):
            reasons.append('outside_plant_azimuth_sector')
        if options['infinite_stem']:
            if metrics['infinite_stem_box_clearance_m']<.005-1e-9:reasons.append('extended_stem_box_overlap')
            if metrics['infinite_stem_second_peduncle_clearance_m']<-1e-9:reasons.append('peduncle_returns_into_stem')
        for reason in reasons:rejected[reason]=rejected.get(reason,0)+1
        if not reasons:
            tomato.generation_info['whole_robot_constraints']={'attempts':attempt,'rejection_counts':rejected,
                'options':options,'metrics':metrics,'scene_version':SCENE_VERSION}
            return tomato,geometry,workspace
    raise RuntimeError(f'工作区/车侧/主茎几何重采样耗尽，拒绝原因：{rejected}')


def robot_reach_bound(model,base_body):
    """所有活动碰撞体距固定B原点的保守上界，不依赖某一次HOME姿态。"""
    if model.body_jntnum[base_body]:raise ValueError('主茎延长证明要求机械臂底座固定')
    distances={int(base_body):0.};reach=0.
    for body in range(base_body+1,model.nbody):
        parent=int(model.body_parentid[body])
        if parent not in distances:continue
        value=distances[parent]+float(np.linalg.norm(model.body_pos[body]))
        for j in np.flatnonzero(model.jnt_bodyid==body):
            # 显式用整数比较，避免NumPy标量与部分MuJoCo版本的枚举在tuple中比较不对称。
            kind=int(model.jnt_type[j])
            if kind in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_BALL)):
                # 绕非原点j旋转：j-Rj，位移最多2||j||。
                value+=2*float(np.linalg.norm(model.jnt_pos[j]))
            elif kind==int(mujoco.mjtJoint.mjJNT_SLIDE):
                if not model.jnt_limited[j]:raise ValueError('无界滑动关节无法建立有限运动范围')
                value+=float(np.max(np.abs(model.jnt_range[j])))
            else:raise ValueError('不支持活动自由基座的主茎延长证明')
        distances[body]=value
    for g in np.flatnonzero(model.geom_contype==1):
        body=int(model.geom_bodyid[g])
        if body not in distances:raise ValueError('活动碰撞体不属于固定机械臂底座子树')
        reach=max(reach,distances[body]+float(np.linalg.norm(model.geom_pos[g]))+float(model.geom_rbound[g]))
    if not np.isfinite(reach) or reach<=0:raise ValueError('未能计算活动碰撞体的运动范围')
    return reach


def add_stem_axis_guard(plant,kp_local,p0_base,radius,reach):
    """圆柱端面放到所有活动碰撞体可达范围之外，实现该整机下的无限主茎。"""
    stem=kp_local['M1']-kp_local['M2'];stem=stem/np.linalg.norm(stem)
    # 对任一机器人几何点x，||x||≤R，故|s·(x-P0)|≤R+|s·P0|。
    # 端面再外推0.1m；在关节范围内不可能越过端面，不是随意指定一根短柱。
    half=float(reach+abs(stem@p0_base)+.1)
    p0=kp_local['P0']
    ET.SubElement(plant,'geom',name=STEM_GUARD,type='cylinder',
        fromto=fmt(np.r_[p0-half*stem,p0+half*stem]),size=str(radius),
        contype='2',conaffinity='1',group='4',rgba='.15 .7 .25 .12')
    return {'name':STEM_GUARD,'robot_reach_bound_m':float(reach),'half_length_m':half,
        'radius_m':float(radius),'axis_base':stem.tolist(),'axis_point_base_m':np.asarray(p0_base).tolist(),
        'viewer_group':4,
        'note':'仅约束活动机械臂/夹爪；端面超出其全部可达范围；默认隐藏，--show-stem-guard显示；静态平台不与虚拟延长段互检。'}


def apply_base_yaw(root,yaw_deg):
    """相对原安装绕机械臂底座+Z转向；不改关节零位或转动整车。

    当前底座竖直安装，底座+Z即世界竖直方向。正式配置目前为180°。
    R_parent_base_new=R_parent_base_old * Rz(yaw)，旋转原点仍是安装座原点。
    """
    if not np.isfinite(yaw_deg):raise ValueError('base_yaw_offset_deg须为有限角度')
    if yaw_deg==0:return
    mount=root.find(".//body[@name='arm_base_mount']")
    if mount is None:raise ValueError('缺少arm_base_mount')
    old=_quat_to_mat(np.fromstring(mount.get('quat','1 0 0 0'),sep=' '))
    angle=np.deg2rad(yaw_deg);c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    mount.set('quat',fmt(_quat_from_mat(old@rotation)))


def mesh_triangles(model, geom):
    """编译网格在主惯性轴系；恢复到geom所属body的局部坐标，保留活动归属。"""
    mesh = model.geom_dataid[geom]
    s,n = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[s:s+n] @ _quat_to_mat(model.geom_quat[geom]).T + model.geom_pos[geom]
    s,n = model.mesh_faceadr[mesh],model.mesh_facenum[mesh]
    return vertices[model.mesh_face[s:s+n]]


def proxy_boxes(triangles, slice_size=None):
    pts = triangles.reshape(-1,3); lo,hi = pts.min(0),pts.max(0)
    if slice_size is None:
        return [(lo,hi)]
    axis = int(np.argmax(hi-lo)); order = [a for a in range(3) if a!=axis]+[axis]
    reverse = np.argsort(order); ordered = triangles[:,:,order]
    cuts = np.linspace(lo[axis],hi[axis],max(1,int(np.ceil((hi[axis]-lo[axis])/slice_size)))+1)
    boxes=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        bounds=slab_bounds(ordered,a,b)
        if bounds is not None:
            boxes.append((bounds[0][reverse],bounds[1][reverse]))
    return boxes


def strip_visual_geometry(root):
    """移除明确不参与碰撞的geom及未引用网格；保留body、关节、site和全部碰撞体。

    仅用于运动学训练的内存优化，不声称与含视觉质量的模型动力学等价。
    必须在build_robot已经显式设置每个geom碰撞属性后调用。
    """
    removed=0
    for node in root.iter():
        for geom in list(node.findall('geom')):
            if geom.get('contype')=='0' and geom.get('conaffinity')=='0':
                node.remove(geom);removed+=1
    used={g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    asset=root.find('asset')
    if asset is not None:
        for mesh in list(asset.findall('mesh')):
            if mesh.get('name') not in used:asset.remove(mesh)
    return removed


def build_robot(config,visuals=True):
    """迁移同门机构。平台关节在零位移移除；机械臂6轴+夹爪3轴仍保留。"""
    root=ET.parse(MODEL).getroot(); original=mujoco.MjModel.from_xml_path(str(MODEL))
    # 候选安装方向仅在显式配置时改变；原始参考文件始终不写回。
    apply_base_yaw(root,float(config.get('base_yaw_offset_deg',0.)))
    # MuJoCo's in-memory XML has no source file to resolve relative mesh paths from.
    root.find('compiler').set('meshdir', str(MODEL.parent))
    tool=root.find(".//body[@name='base_link']"); tool.set('name','gripper_base')
    root.remove(root.find('keyframe')); root.remove(root.find('contact'))
    contact=ET.SubElement(root,'contact')
    for pair in sorted([sorted(x) for x in ADJACENT]):
        ET.SubElement(contact,'exclude',body1=pair[0],body2=pair[1])
    keep=set(ARM)|{'gear_joint','right_finger_joint','left_finger_joint'}
    for body in root.iter('body'):
        for joint in list(body.findall('joint')):
            if joint.get('name') not in keep:
                body.remove(joint)
    actuators=root.find('actuator')
    for actuator in list(actuators):
        if actuator.get('joint') not in keep:
            actuators.remove(actuator)
    # 不依赖contype默认值或视觉分组；明确划分1机器人、4平台/地面。
    arm_bodies={'shoulder_Link','upperArm_Link','foreArm_Link','wrist1_Link','wrist2_Link','wrist3_Link'}
    counts={'tool_boxes':0,'platform_boxes':0}; coverage=[]
    for body in root.iter('body'):
        name=body.get('name'); moving=name in TOOL or name in arm_bodies
        for geom in list(body.findall('geom')):
            gid=obj(original,mujoco.mjtObj.mjOBJ_GEOM,geom.get('name'))
            if name in arm_bodies:
                physical=bool(original.geom_contype[gid] or original.geom_conaffinity[gid])
                geom.set('contype','1' if physical else '0');geom.set('conaffinity','7' if physical else '0')
                continue
            # 底盘/支架/相机/工具统一拆出可见网格与碰撞代理；安装座本身也有包络。
            geom.set('contype','0');geom.set('conaffinity','0');geom.set('group','1')
            if not (original.geom_contype[gid] or original.geom_conaffinity[gid]):
                continue
            if original.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                # 升降柱和篮筐已有box/cylinder代理，保留其空腔/尺寸，不再取整件盒。
                geom.set('contype','1' if moving else '4');geom.set('conaffinity','7' if moving else '1')
                geom.set('group','3')
                continue
            triangles=mesh_triangles(original,gid)
            sliced=name in {'right_finger_link','left_finger_link'} and ('mesh_2' in geom.get('name') or 'mesh_1' in geom.get('name'))
            bounds=proxy_boxes(triangles,config['finger_slice_m'] if sliced else None)
            for k,(lo,hi) in enumerate(bounds):
                half=np.maximum((hi-lo)/2,1e-6)
                ET.SubElement(body,'geom',name=f"proxy_{geom.get('name')}_{k}",type='box',
                    pos=fmt((hi+lo)/2),size=fmt(half),group='3',contype='1' if moving else '4',
                    conaffinity='7' if moving else '1',rgba='0 0.8 1 0.22' if moving else '0.8 0.65 0.1 0.14')
                counts['tool_boxes' if moving else 'platform_boxes']+=1
            coverage.append({'source_geom':geom.get('name'),'body':name,'boxes':len(bounds)})
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='fixed_ground',type='plane',size='3 3 .05',rgba='.25 .28 .3 1',contype='4',conaffinity='1')
    # 保留现有固定轴向；TCP位置由gripper.yaml统一读取，不在此写死工具长度。
    tcp_config=yaml.safe_load((CONFIG_DIR/'gripper.yaml').read_text())
    wrist=root.find(".//body[@name='wrist3_Link']")
    tcp=ET.SubElement(wrist,'body',name='tool_tcp',pos=fmt(tcp_config['tcp_position']),quat=fmt(tcp_config['tcp_quaternion']))
    ET.SubElement(tcp,'site',name='grasp_tcp',type='sphere',size='.003',rgba='1 .8 0 1')
    for axis,end,color in [('x',[.03,0,0],'1 0 0 1'),('y',[0,.03,0],'0 1 0 1'),('z',[0,0,.03],'0 0 1 1')]:
        ET.SubElement(tcp,'site',name='tcp_'+axis,type='capsule',size='.0007',fromto='0 0 0 '+fmt(end),rgba=color)
    removed=0 if visuals else strip_visual_geometry(root)
    return root,{'proxy_counts':counts,'proxy_sources':coverage,'tcp':tcp_config,'platform_fixed':True,
                 'finger_slice_m':float(config['finger_slice_m']),
                 'base_yaw_offset_deg':float(config.get('base_yaw_offset_deg',0.)),
                 'visual_meshes_loaded':bool(visuals),'removed_visual_geoms':removed}


def geom_distance(model,data,a,b,distmax=1e-6):
    """几何距离；补足两个盒子中心重合时距离查询返回0的退化情况。

    OBB分离轴定理：两盒的6个面法向和9个边叉积共15个候选轴。
    若每个轴投影都严格重叠，则盒子穿透；任一轴分离则不穿透。
    只在box/box查询接近0时补查，不把包围球重叠误当实际碰撞。
    """
    distance=float(mujoco.mj_geomDistance(model,data,int(a),int(b),distmax,None))
    if (abs(distance)<=1e-7 and int(model.geom_type[a])==int(mujoco.mjtGeom.mjGEOM_BOX)
            and int(model.geom_type[b])==int(mujoco.mjtGeom.mjGEOM_BOX)):
        A=data.geom_xmat[a].reshape(3,3);B=data.geom_xmat[b].reshape(3,3)
        axes=[*A.T,*B.T,*[np.cross(x,y) for x in A.T for y in B.T]]
        delta=data.geom_xpos[b]-data.geom_xpos[a];separations=[]
        for axis in axes:
            norm=np.linalg.norm(axis)
            if norm<1e-10:continue
            axis=axis/norm
            separations.append(abs(delta@axis)-np.abs(axis@A)@model.geom_size[a]-np.abs(axis@B)@model.geom_size[b])
        overlap=max(separations)
        if overlap<-1e-7:distance=float(overlap)
    return distance


class SceneCollision:
    """检查机械臂自碰撞、车体和植物，排除相邻安装件的设计接触。"""
    def __init__(self, model):
        self.model = model
        moving = np.flatnonzero(model.geom_contype == 1)
        static = np.flatnonzero(np.isin(model.geom_contype, [2, 4]))
        pairs = []
        for ii,a in enumerate(moving):
            for b in np.r_[moving[ii+1:],static]:
                ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
                na,nb=self.body(ba),self.body(bb)
                if ba==bb or frozenset((na,nb)) in ADJACENT:
                    continue
                pairs.append((int(a),int(b)))
        self.pairs=np.asarray(pairs,dtype=int).reshape(-1,2)
        self.pair_set = {tuple(sorted(pair)) for pair in pairs}
        self.radii=model.geom_rbound[self.pairs].sum(1)
        # plane的rbound=0，不能用球排除。
        self.planes=np.any(model.geom_type[self.pairs]==mujoco.mjtGeom.mjGEOM_PLANE,axis=1)
        self.stem_guard=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,STEM_GUARD)
        self.guard_pairs=np.any(self.pairs==self.stem_guard,axis=1) if self.stem_guard>=0 else np.zeros(len(self.pairs),bool)

    def body(self,idx):
        return mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,idx) or 'world'

    def _collision(self, a, b, distance):
        types = {int(self.model.geom_contype[g]) for g in (a, b)}
        category = 'plant' if 2 in types else ('vehicle_or_ground' if 4 in types else 'self')
        return {
            'pair': [self.body(self.model.geom_bodyid[a]), self.body(self.model.geom_bodyid[b])],
            'geoms': [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (a, b)],
            'distance_m': float(distance),
            'category': category,
        }

    def contacts(self, data, margin=0., *, include_mujoco=False):
        if include_mujoco:
            # mj_step留下积分前的求解器接触，下面的距离查询复核积分后的几何。
            native_hits = {}
            for contact in data.contact:
                pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                if pair in self.pair_set and contact.dist < -1e-7:
                    distance = min(contact.dist, native_hits.get(pair, 0.))
                    native_hits[pair] = distance
            if native_hits:
                return [self._collision(*pair, distance) for pair, distance in native_hits.items()]
        a,b=self.pairs.T
        possible=self.planes | (np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)<=self.radii+max(margin,0))
        if self.guard_pairs.any():
            # 长柱的大包围球太宽；用到轴线的径向距离保守剔除，减少每步精确查询。
            pairs=self.pairs[self.guard_pairs]
            other=np.where(pairs[:,0]==self.stem_guard,pairs[:,1],pairs[:,0])
            axis=data.geom_xmat[self.stem_guard].reshape(3,3)[:,2]
            relative=data.geom_xpos[other]-data.geom_xpos[self.stem_guard]
            radial=relative-np.outer(relative@axis,axis)
            possible[self.guard_pairs]=np.linalg.norm(radial,axis=1)<=self.model.geom_rbound[other]+self.model.geom_size[self.stem_guard,0]+max(margin,0)
        result=[]
        for x,y in self.pairs[possible]:
            distance=geom_distance(self.model,data,int(x),int(y),max(margin,0)+1e-6)
            if distance<margin-1e-7:
                result.append(self._collision(x, y, distance))
        return result


def robot_from_snapshot(config,directory,visuals):
    """复用本项目已导出的机器人碰撞代理，重建植物；低内存任务不重复编译原CAD。

    只复用固定平台/工具/机械臂，原番茄、标记与home关键帧删除后由当前配置重建。
    安装和TCP必须匹配；无视觉网格的快照不能假装恢复完整CAD显示。
    """
    directory=Path(directory);manifest=json.loads((directory/'scene.json').read_text())
    from task_joint_limits import check_snapshot_limit_profile
    check_snapshot_limit_profile(manifest,config)
    if manifest.get('harvest_cycle_model'):raise ValueError('请选择采摘前机器人快照，不复用已经拆分果穗的任务场景')
    if manifest.get('mocap_backend') != config['pose_controller']:
        raise ValueError('快照的 mocap 执行器配置与当前配置不一致')
    if manifest.get('scene_version')!=SCENE_VERSION:raise ValueError('机器人快照版本不匹配')
    tcp=yaml.safe_load((CONFIG_DIR/'gripper.yaml').read_text())
    saved_tcp = manifest.get('tcp', {})
    pose_keys = ('tcp_position', 'tcp_quaternion')
    if (any(saved_tcp.get(key) != tcp[key] for key in pose_keys)
            or manifest.get('base_yaw_offset_deg') != config.get('base_yaw_offset_deg', 0.)):
        raise ValueError('机器人快照的TCP或底座安装关系与当前配置不一致')
    saved_slice=manifest.get('finger_slice_m')
    if saved_slice is None:
        saved_config=directory.parent/'config.yaml'
        if not saved_config.is_file():raise ValueError('旧机器人快照缺少手指分片参数和配套config.yaml，请重新导出')
        saved_slice=yaml.safe_load(saved_config.read_text()).get('finger_slice_m')
    if saved_slice!=config.get('finger_slice_m',.004):
        raise ValueError('机器人快照的手指碰撞分片与当前配置不一致')
    if visuals and not manifest.get('visual_meshes_loaded',True):raise ValueError('快照不含视觉网格；请使用collision-only或重新构建')
    root=ET.parse(directory/'scene.xml').getroot();world=root.find('worldbody')
    root.find('compiler').set('meshdir', str(MODEL.parent))
    for name in ('bunch','hopp_base_frame'):
        element=world.find(f"body[@name='{name}']")
        if element is not None:world.remove(element)
    for key in list(root.findall('keyframe')):root.remove(key)
    manifest['robot_snapshot']=repo_relative(directory)
    manifest['finger_slice_m']=float(saved_slice)
    if not visuals:strip_visual_geometry(root)
    manifest['visual_meshes_loaded']=bool(visuals)
    return root,manifest


class WholeRobotScene:
    def __init__(self,scene_seed=37,config=None,*,visuals=True,robot_snapshot=None,pose_seed=None):
        self.config=copy.deepcopy(config or load_config()); self.scene_seed=scene_seed
        # 参数拼写/范围错误应先报出，避免为无效配置加载完整整车网格。
        options=plant_constraint_options(self.config)
        hemisphere=harvest_hemisphere(self.config)
        self.pregrasp_parameters=pregrasp_options(self.config)
        root,self.manifest=(build_robot(self.config,visuals=visuals) if robot_snapshot is None
                            else robot_from_snapshot(self.config,robot_snapshot,visuals))
        from mocap_pose_controller import strip_mocap_nodes, configure_mocap_xml
        strip_mocap_nodes(root)
        # 同门基座位于整车世界约z=.770m。番茄先在基座系生成，再随基座刚体变换。
        bare=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode')); bd=mujoco.MjData(bare);mujoco.mj_forward(bare,bd)
        from task_joint_limits import (apply_task_joint_limits,reference_limits_for_profile,
                                       update_position_actuator_ranges)
        reference=self.manifest.get('reference_arm_joint_limits_rad')
        if reference is None:
            reference=bare.jnt_range[[obj(bare,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ARM]].tolist()
        profile=self.config.get('robot_joint_limit_profile')
        if profile is not None:
            # 在任务限位之前选择型号范围；旧快照只作为几何来源，不覆盖其文件。
            self.manifest.setdefault('source_arm_joint_limits_rad',copy.deepcopy(reference))
            reference=reference_limits_for_profile(ARM,reference,profile).tolist()
            self.manifest['robot_joint_limit_profile']=profile
        effective=apply_task_joint_limits(root,ARM,reference,
            self.config.get('task_joint_limits_deg',{}),self.config['start_joints_deg'])
        if profile is not None:update_position_actuator_ranges(root,ARM,effective)
        # 保留原始范围，下一次从快照换任务限位时仍可恢复模型参考范围。
        self.manifest['reference_arm_joint_limits_rad']=reference
        self.manifest['task_joint_limits_deg']=copy.deepcopy(self.config.get('task_joint_limits_deg',{}))
        self.manifest['effective_arm_joint_limits_deg']=np.rad2deg(effective).tolist()
        base=obj(bare,mujoco.mjtObj.mjOBJ_BODY,'arm_base_mount')
        self.base_position=bd.xpos[base].copy(); self.base_rotation=bd.xmat[base].reshape(3,3).copy()
        self.tomato,self.geometry,workspace=sample_task_tomato(scene_seed,self.config)
        self.pose_seed = pose_seed
        if pose_seed is not None:
            from paper_dataset import randomize_model_pose
            self.geometry = randomize_model_pose(self.tomato, self.geometry, self.config, pose_seed)
        self.manifest['scene_version']=SCENE_VERSION
        self.manifest['plant_constraints']=options
        self.manifest['harvest_hemisphere_direction_base']=hemisphere.tolist()
        _,self.pregrasp_info=pregrasp_candidates(self.geometry,self.pregrasp_parameters)
        self.manifest['pregrasp_sector']=self.pregrasp_info
        self.workspace_info=None
        if workspace is not None:
            bounds = workspace_bounds(workspace)
            contain_bunch = workspace.get('contain_bunch', False)
            self.workspace_info={'configuration':copy.deepcopy(workspace),'bounds_base_m':bounds.tolist(),
                'grasp_point_base_m':self.geometry.grasp_point.tolist(),
                'grasp_point_world_m':(self.base_position+self.base_rotation@self.geometry.grasp_point).tolist(),
                'note':('限制Pg、果梗和果穗盒，主茎为周围障碍。' if contain_bunch else '仅限制Pg。')
                       + '此边界用于目标放置，路径可行性取决于机器人位姿和碰撞。'}
        segment_type = self.config.get('plant_geometry', {}).get('segment_type', 'capsule')
        if segment_type not in ('capsule', 'cylinder'):
            raise ValueError('plant_geometry.segment_type须为cylinder或capsule')
        plant = self.tomato.plant_body(segment_type)
        if options['infinite_stem']:
            reach=robot_reach_bound(bare,base)
            self.manifest['stem_axis_guard']=add_stem_axis_guard(plant,self.tomato.kp_local,
                self.tomato.kp['P0'],self.tomato.cfg['main_stem']['diameter']/2,reach)
        plant.set('pos',fmt(self.base_position+self.base_rotation@self.tomato.placement))
        plant.set('quat',fmt(_quat_from_mat(self.base_rotation)))
        root.find('worldbody').append(plant)
        # HOPP计算在基座系，标记整体放到基座固定坐标体中，避免重复/遗漏变换。
        marker_root=ET.Element('mujoco');ET.SubElement(marker_root,'worldbody')
        add_hopp_markers(marker_root,self.geometry)
        # HOPP平面与隐藏主茎圆柱分组独立；显示圆柱开关不再控制HOPP平面。
        marker_root.find("worldbody/geom[@name='hopp_plane']").set('group','2')
        add_pregrasp_sector_markers(marker_root,self.geometry,self.pregrasp_parameters)
        marker=ET.SubElement(root.find('worldbody'),'body',name='hopp_base_frame',pos=fmt(self.base_position),quat=fmt(_quat_from_mat(self.base_rotation)))
        for element in marker_root.find('worldbody'):
            marker.append(element)
        if self.workspace_info is not None:
            # 青色12条边显示目标作业区；site仅用于显示。
            bounds=np.asarray(self.workspace_info['bounds_base_m'])
            corners=[bounds[np.arange(3),[(k>>axis)&1 for axis in range(3)]] for k in range(8)]
            for k in range(8):
                for axis in range(3):
                    other=k^(1<<axis)
                    if k<other:
                        ET.SubElement(marker,'site',name=f'pg_workspace_{k}_{other}',type='capsule',
                            fromto=fmt(np.r_[corners[k],corners[other]]),size='.001',rgba='0 .85 1 .5',group='5')
        # 安装变换和主茎范围已拷贝完成，最终编译前释放临时机器人，避免两套网格同时驻留。
        if self.config['pose_controller']['mode'] != 'mocap_weld':
            raise ValueError('此仿真环境需要 mocap_weld 执行器')
        for joint,angle in zip(ARM,np.deg2rad(self.config['start_joints_deg'])):
            bd.qpos[addr(bare,joint)]=angle
        mujoco.mj_forward(bare,bd)
        tcp=obj(bare,mujoco.mjtObj.mjOBJ_BODY,'tool_tcp')
        configure_mocap_xml(root,self.config,bd.xpos[tcp],bd.xmat[tcp].reshape(3,3))
        self.manifest['mocap_backend']=copy.deepcopy(self.config['pose_controller'])
        del bare,bd
        self.xml=ET.tostring(root,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.arm_qadr=np.asarray([addr(self.model,n) for n in ARM]);self.arm_dadr=np.asarray([self.model.jnt_dofadr[obj(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ARM])
        self.arm_jids=np.asarray([obj(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ARM])
        self.gripper_qadr=np.asarray([addr(self.model,n) for n in ('gear_joint','right_finger_joint','left_finger_joint')])
        self.tcp_id=obj(self.model,mujoco.mjtObj.mjOBJ_BODY,'tool_tcp')
        tool_ids = [obj(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in sorted(TOOL)]
        self.manifest['gripper_mass_kg'] = float(self.model.body_mass[tool_ids].sum())
        self.checker = SceneCollision(self.model)
        self.set_state(np.deg2rad(self.config['start_joints_deg']),self.config['open_gear_rad'])
        self.start_q=self.data.qpos.copy()
        self.start_ctrl=self.data.ctrl.copy()

    def set_state(self,arm=None,gear=None):
        if arm is not None:self.data.qpos[self.arm_qadr]=arm
        if gear is not None:
            gear_joint=obj(self.model,mujoco.mjtObj.mjOBJ_JOINT,'gear_joint')
            if not np.isfinite(gear) or not self.model.jnt_range[gear_joint,0]<=gear<=self.model.jnt_range[gear_joint,1]:
                raise ValueError('gear超出参考关节范围')
            self.data.qpos[addr(self.model,'gear_joint')]=gear
            self.data.qpos[addr(self.model,'right_finger_joint')]=-.020995*gear
            self.data.qpos[addr(self.model,'left_finger_joint')]=-.020999*gear
        self.data.qvel[:]=0
        # 为HOME/reset/导出保留一致ctrl；正常mocap执行在独立后端调用mj_step。
        for name,joint in [('gear_servo','gear_joint')]+[(f'j{i+1}',n) for i,n in enumerate(ARM)]:
            actuator=obj(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
            self.data.ctrl[actuator]=self.data.qpos[addr(self.model,joint)]
        mujoco.mj_forward(self.model,self.data)

        # 仅在显式设姿态/reset时同步虚拟目标；正常mocap执行不调用set_state。
        target=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'drl_mocap_target')
        if target>=0:
            mid=self.model.body_mocapid[target]
            self.data.mocap_pos[mid]=self.data.xpos[self.tcp_id]
            self.data.mocap_quat[mid]=_quat_from_mat(self.data.xmat[self.tcp_id].reshape(3,3))
            mujoco.mj_forward(self.model,self.data)

    def tcp_pose(self):
        position=self.base_rotation.T@(self.data.xpos[self.tcp_id]-self.base_position)
        rotation=self.base_rotation.T@self.data.xmat[self.tcp_id].reshape(3,3)
        return position,rotation

    def export(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        root=ET.fromstring(self.xml)
        meshdir = Path(os.path.relpath(MODEL.parent, directory.resolve())).as_posix()
        root.find('compiler').set('meshdir', meshdir)
        keys=ET.SubElement(root,'keyframe')
        ET.SubElement(keys,'key',name='home',qpos=fmt(self.start_q),ctrl=fmt(self.start_ctrl))
        (directory/'scene.xml').write_text(ET.tostring(root,encoding='unicode'))
        info=copy.deepcopy(self.manifest)
        info.update(scene_seed=self.scene_seed,base_position_m=self.base_position.tolist(),base_rotation=self.base_rotation.tolist(),
            start_qpos=self.start_q.tolist(),hopp=self.geometry.describe(self.tomato.cfg['main_stem']['diameter']/2),
            scene_generation=self.tomato.generation_info,target_workspace=self.workspace_info)
        (directory/'scene.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
