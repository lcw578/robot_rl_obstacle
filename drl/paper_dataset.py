"""500份固定数字模型及400/100隔离；不把生成模型称为作者的RGB-D重建数据。"""
import hashlib
import json
from pathlib import Path
from project_paths import PROJECT_ROOT

DATASET_VERSION='generated_tomato_400_100_v1'


def manifest_path(config):
    value=config.get('paper_training',{}).get('dataset_manifest')
    if value is None:return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ('assets', 'datasets'):
        # Old training configs resolved this prefix relative to drl/.
        path = Path(*path.parts[1:])
        current = PROJECT_ROOT / path
        if current.is_file():
            return current
        archived = PROJECT_ROOT / 'archive' / path
        if archived.is_file():
            return archived
        return current
    return PROJECT_ROOT / path


def read_manifest(config):
    path=manifest_path(config)
    if path is None:return None
    raw=path.read_bytes()
    expected=config.get('paper_training',{}).get('dataset_manifest_sha256')
    if expected and hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('数据集manifest与训练检查点记录不一致')
    data=json.loads(raw)
    if data.get('version')!=DATASET_VERSION or not data.get('complete'):
        raise ValueError('数字模型集未完整生成，请先运行prepare_paper_dataset.py')
    records=data.get('models',[])
    seeds=[r['scene_seed'] for r in records]
    train=data.get('training_seeds',[]);test=data.get('test_seeds',[])
    if (len(records)!=500 or len(train)!=400 or len(test)!=100
            or len(set(seeds))!=500 or len(set(train))!=400 or len(set(test))!=100
            or set(train)&set(test) or set(seeds)!=set(train)|set(test)
            or any(type(s) is not int or s<0 for s in seeds)):
        raise ValueError('数据集须包含500份唯一模型，400训练/100测试且不重叠')
    if any(Path(r['file']).is_absolute() or '..' in Path(r['file']).parts for r in records):
        raise ValueError('模型文件必须位于manifest相对目录内')
    if any(r['split']!=('training' if r['scene_seed'] in train else 'test') for r in records):
        raise ValueError('模型split标签与400/100清单矛盾')
    if len({r.get('geometry_sha256') for r in records})!=500:
        raise ValueError('500份模型的几何/位置签名重复或缺失')
    return path,data


def load_digital_model(scene_seed,config):
    """直接装载已冻结的几何，不依赖当前生成器重新抽样；一次只读一份。"""
    path,manifest=read_manifest(config)
    entry=next((r for r in manifest['models'] if r['scene_seed']==scene_seed),None)
    if entry is None:raise ValueError(f'seed={scene_seed}不属于本数字模型集')
    raw=(path.parent/entry['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
        raise ValueError('数字模型文件哈希不符，禁止静默再生成')
    record=json.loads(raw)
    if record['scene_seed']!=scene_seed:raise ValueError('模型编号与文件内容不符')
    import numpy as np
    from env_cluster import TomatoClusterScene
    from hopp import compute_hopp
    from whole_robot_scene import bunch_bounds, harvest_hemisphere, workspace_bounds
    tomato=TomatoClusterScene();tomato.cfg=record['plant_config']
    tomato.kp={k:np.asarray(v,float) for k,v in record['keypoints_base_m'].items()}
    tomato.kp_local={k:np.asarray(v,float) for k,v in record['keypoints_local_m'].items()}
    tomato.placement=np.asarray(record['placement_base_m'],float)
    tomato._box={k:np.asarray(v,float) for k,v in record['box_local'].items()}
    tomato.generation_info=record['generation_info']
    geometry=compute_hopp(tomato.kp,tomato.cfg['grasp']['d_stem'],
                          tomato.cfg['grasp']['zone_r'],harvest_hemisphere(config))
    # 模型位置/形态固定；修改采样范围应创建新数据集，不把旧模型伪装成新分布。
    for key in ('target_workspace','plant_constraints','harvest_region','plant_geometry'):
        if config.get(key)!=manifest['scene_config'].get(key):
            raise ValueError('数据集与当前'+key+'不同，请独立生成新数据集')
    workspace = config.get('target_workspace')
    if workspace is not None:
        bounds = workspace_bounds(workspace)
        occupied = (bunch_bounds(tomato, geometry.grasp_point) if workspace.get('contain_bunch', False)
                    else geometry.grasp_point[:, None])
        if np.any(occupied[:, 0] < bounds[:, 0] - 1e-9) or np.any(occupied[:, -1] > bounds[:, 1] + 1e-9):
            raise ValueError(f'数字模型 {scene_seed} 超出目标工作区')
    return tomato,geometry,workspace


def randomize_model_pose(tomato, geometry, config, pose_seed):
    """刚体变换数字模型，保留主茎、果梗、果穗之间的局部几何。"""
    import numpy as np
    from env_cluster import _quat_from_mat, _quat_to_mat
    from hopp import compute_hopp
    from whole_robot_scene import harvest_hemisphere, workspace_bounds

    workspace = config['target_workspace']
    if workspace.get('contain_bunch', False):
        raise ValueError('回合刚体随机化以Pg为作业区参考')
    maximum = float(config['paper_training']['rigid_rotation_deg'])
    if not np.isfinite(maximum) or not 0 < maximum <= 180:
        raise ValueError('rigid_rotation_deg须在(0, 180]内')
    rng = np.random.default_rng(pose_seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    half_angle = np.deg2rad(rng.uniform(0., maximum)) / 2
    rotation = _quat_to_mat(np.r_[np.cos(half_angle), np.sin(half_angle) * axis])
    bounds = workspace_bounds(workspace)
    target = rng.uniform(bounds[:, 0], bounds[:, 1])
    local_grasp = geometry.grasp_point - tomato.placement
    tomato.placement = target - rotation @ local_grasp
    tomato.kp_local = {key: rotation @ point for key, point in tomato.kp_local.items()}
    tomato.kp = {key: point + tomato.placement for key, point in tomato.kp_local.items()}
    tomato._box['center'] = rotation @ tomato._box['center']
    tomato._box['quat'] = _quat_from_mat(rotation @ _quat_to_mat(tomato._box['quat']))
    tomato.generation_info['episode_pose'] = {'seed': int(pose_seed), 'angle_deg': float(np.rad2deg(2 * half_angle))}
    return compute_hopp(tomato.kp, tomato.cfg['grasp']['d_stem'],
                        tomato.cfg['grasp']['zone_r'], harvest_hemisphere(config))
