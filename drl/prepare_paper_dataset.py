"""用户执行：冻结500份参数化几何、400训练/100测试；不加载整机、不训练。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from project_paths import DEFAULT_CONFIG


def generate(args):
    import numpy as np
    import yaml
    from whole_robot_scene import sample_task_tomato
    from paper_dataset import DATASET_VERSION
    cfg=yaml.safe_load(args.config.read_text())
    cfg.get('paper_training',{}).pop('dataset_manifest',None)  # 生成阶段才调用采样器。
    seeds=list(range(args.first_seed,args.first_seed+500))
    order=seeds.copy();random.Random(args.split_seed).shuffle(order)
    # 锚点固定留在训练集；不因模型是否易达筛选训练/测试。
    anchor=order.index(args.first_seed)
    if anchor>=400:order[0],order[anchor]=order[anchor],order[0]
    train,test=order[:400],order[400:]
    data={'version':DATASET_VERSION,'complete':False,
          'provenance':'parameterized_generated_surrogate_NOT_author_RGBD_dataset',
          'training_seeds':train,'test_seeds':test,'split_seed':args.split_seed,
          'training_anchor_seed':args.first_seed,
          'scene_config':{k:cfg.get(k) for k in ('target_workspace','plant_constraints','harvest_region','plant_geometry')},
          'models':[],'generation_errors':[]}
    models=args.output/'models';models.mkdir(exist_ok=True)
    def json_default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    for index,seed in enumerate(seeds):
        try:
            tomato,geometry,_=sample_task_tomato(seed,cfg)
            record={'scene_seed':seed,'plant_config':tomato.cfg,
                'keypoints_base_m':tomato.kp,'keypoints_local_m':tomato.kp_local,
                'placement_base_m':tomato.placement,'box_local':tomato._box,
                'grasp_point_base_m':geometry.grasp_point,'generation_info':tomato.generation_info}
            raw=json.dumps(record,default=json_default,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
            geometry_raw=json.dumps({k:v for k,v in record.items() if k not in ('scene_seed','generation_info')},
                default=json_default,sort_keys=True,allow_nan=False).encode()
            geometry_hash=hashlib.sha256(geometry_raw).hexdigest()
            if any(r['geometry_sha256']==geometry_hash for r in data['models']):
                raise ValueError('生成模型的几何/位置重复，未用新seed替换')
            name=f'model_{seed}.json';(models/name).write_bytes(raw)
            data['models'].append({'scene_seed':seed,'file':'models/'+name,
                'split':'training' if seed in train else 'test',
                'sha256':hashlib.sha256(raw).hexdigest(),'geometry_sha256':geometry_hash})
        except (ValueError,RuntimeError) as exc:
            data['generation_errors'].append({'scene_seed':seed,'error':str(exc)})
        if (index+1)%25==0:print('几何生成',index+1,'/500，失败',len(data['generation_errors']),flush=True)
    data['complete']=len(data['models'])==500 and not data['generation_errors']
    (args.output/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (args.output/'generation_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    print('400/100模型集：',args.output/'manifest.json','；完整：',data['complete'],flush=True)
    if not data['complete']:raise ValueError('模型集不完整；失败编号已保留，未换种子凑足500份')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-seed',type=int,default=1000000)
    p.add_argument('--split-seed',type=int,default=20261110)
    args=p.parse_args()
    if args.first_seed<0:p.error('first-seed须非负')
    if os.environ.get('TOMATO_DATASET_GUARDED')!='1':
        from resource_guard import guarded_run
        args.output.mkdir(parents=True,exist_ok=False)
        environment=os.environ.copy();environment['TOMATO_DATASET_GUARDED']='1'
        with (args.output/'console.log').open('w') as log:
            result=guarded_run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
                log,Path.cwd(),environment,timeout=1800,memory_mb=1536,reserve_mb=2048,
                cores=1,address_space_mb=8192)
        (args.output/'resource_report.json').write_text(json.dumps(result,indent=2))
        print('数据集生成状态：',result,flush=True)
        if result['exit_code']!=0 or result['resource_status']!='completed':raise SystemExit(1)
    else:generate(args)


if __name__=='__main__':main()
