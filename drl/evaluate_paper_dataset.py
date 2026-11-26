"""用户执行的100模型独立最终测试；测试结果不回写best或经验池。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

from project_paths import repo_relative


def evaluate_models(model,factory,seeds,settings,on_model=None):
    """独立评估固定模型，起点冲突和运动失败均计入总分母。"""
    rows=[]
    for seed in seeds:
        row={'scene_seed':seed,'attempts':[],'success':False,'initialization_error':None}
        try:env=factory(seed)
        except (ValueError,RuntimeError) as exc:
            row['initialization_error']=str(exc);rows.append(row)
            if on_model:on_model(rows)
            continue
        try:
            attempt_limit = 2 if settings.get('collision_retry', True) else 1
            for attempt in range(1, attempt_limit + 1):
                try:
                    obs,initial=env.reset(seed=seed)
                except (ValueError, RuntimeError) as exc:
                    row['initialization_error'] = str(exc)
                    break
                path_length=0.
                previous_e=float(initial['distance_m']);previous_angle=float(initial['axis_error_rad'])
                unchanged=0
                for step in range(env.cfg['max_episode_steps']):
                    action=model.predict(obs,deterministic=True)[0]
                    obs,reward,term,trunc,info=env.step(action)
                    path_length += info['tcp_substep_path_length_m']
                    success=(info['distance_m']<settings['position_tolerance_m']
                        and info['axis_error_rad']<settings['axis_tolerance_rad']
                        and not info['collision'] and not info.get('joint_limit',False)
                        and not info.get('execution_failure'))
                    stopped=(abs(info['distance_m']-previous_e)<settings['stagnation_distance_delta_m']
                        and abs(info['axis_error_rad']-previous_angle)<settings['stagnation_angle_delta_rad'])
                    unchanged=unchanged+1 if stopped else 0
                    previous_e=float(info['distance_m']);previous_angle=float(info['axis_error_rad'])
                    if success or term or trunc or unchanged>=settings['stagnation_consecutive_steps']:break
                if success:reason='success'
                elif info.get('episode_end_reason'):reason=info['episode_end_reason']
                elif unchanged>=settings['stagnation_consecutive_steps']:reason='stagnation'
                else:reason='test_horizon'
                row['attempts'].append({'attempt':attempt,'steps':step+1,'success':bool(success),
                    'platform_success_at_end':bool(info.get('is_success')),
                    'distance_m':float(info['distance_m']),'axis_error_rad':float(info['axis_error_rad']),
                    'end_reason':reason,'collision':bool(info['collision']),
                    'joint_limit':bool(info.get('joint_limit',False)),
                    'execution_failure':bool(info.get('execution_failure')),'contacts':info.get('contacts',[]),
                    'tcp_path_length_m':path_length,
                    'initial_shortest_path_m':float(initial['distance_m']),
                    'path_excess_m':path_length-float(initial['distance_m'])})
                if success:row['success']=True;break
                if not info['collision']:break
        finally:env.close()
        rows.append(row)
        if on_model:on_model(rows)
    successful=sum(r['success'] for r in rows)
    last=[r['attempts'][-1] for r in rows if r['attempts']]
    def moments(key):
        values=[r[key] for r in last if r[key] is not None]
        return {'count':len(values),'mean':statistics.mean(values) if values else None,
                'std_population':statistics.pstdev(values) if values else None}
    return {'split':'held_out_test_NOT_for_model_selection','model_count':len(rows),
            'successful_models':successful,'success_rate':successful/len(rows),
            'attempt_count':sum(len(r['attempts']) for r in rows),
            'initialization_errors':sum(r['initialization_error'] is not None for r in rows),
            'evaluated_model_count':len(last),
            'path_sampling':'retained_actual_tcp_at_physics_substeps',
            'path_excess_definition':'tcp_path_length_m - initial_shortest_path_m (signed, including failures)',
            'metrics':{key:moments(key) for key in ('distance_m','axis_error_rad','steps','path_excess_m')},
            'scenes':rows}


def run(args):
    import yaml
    from paper_sac import PaperSAC
    from paper_method import PaperPoseGoalEnv,check_policy_fusion_compatibility
    from paper_dataset import read_manifest
    cfg=yaml.safe_load((args.policy.parent/'config.yaml').read_text())
    path,dataset=read_manifest(cfg)
    snapshot = args.policy.parent / "scene"
    factory=lambda seed:PaperPoseGoalEnv(seed,config=cfg,visuals=False,robot_snapshot=snapshot)
    template=factory(cfg['scene_seed'])
    try:
        try:
            model = PaperSAC.load(args.policy, env=template, device='cpu')
        except ValueError as exc:
            if str(exc).startswith('Observation spaces do not match'):
                raise ValueError('检查点观测空间与当前环境不匹配。旧版策略须按新版观测重新训练。') from exc
            raise
        check_policy_fusion_compatibility(model,cfg)
        settings=cfg['paper_training']['final_test']
        def partial(rows):
            (args.output/'test_partial.json').write_text(json.dumps({'completed_models':len(rows),
                'planned_models':100,'scenes':rows},ensure_ascii=False,indent=2))
            print('测试进度',len(rows),'/100，成功',sum(r['success'] for r in rows),flush=True)
        result=evaluate_models(model,factory,dataset['test_seeds'],settings,partial)
        result.update(policy=repo_relative(args.policy),policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                      dataset_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      dataset_provenance=dataset.get('provenance'),
                      thresholds=settings,collision_retry=settings.get('collision_retry', True))
        (args.output/'test_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('独立测试结果：',result['successful_models'],'/100；报告：',args.output/'test_report.json',flush=True)
    finally:template.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--memory-budget-mb',type=int,default=2048)
    p.add_argument('--memory-reserve-mb',type=int,default=1024,help='桌面保留MiB，沿用用户确认的1GiB')
    args=p.parse_args()
    if min(args.memory_budget_mb,args.memory_reserve_mb)<=0:p.error('内存预算/预留须为正整数MiB')
    if os.environ.get('TOMATO_FINAL_TEST_GUARDED')!='1':
        from resource_guard import guarded_run,startup_memory_message
        args.output.mkdir(parents=True,exist_ok=False)
        environment=os.environ.copy();environment['TOMATO_FINAL_TEST_GUARDED']='1'
        with (args.output/'console.log').open('w') as log:
            result=guarded_run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],log,
                Path.cwd(),environment,timeout=10800,memory_mb=args.memory_budget_mb,reserve_mb=args.memory_reserve_mb,cores=1,address_space_mb=16384)
        (args.output/'resource_report.json').write_text(json.dumps(result,indent=2))
        print('独立测试运行状态：',result,flush=True)
        if result['resource_status']=='not_started_low_memory':print(startup_memory_message(result))
        if result['exit_code']!=0 or result['resource_status']!='completed':raise SystemExit(1)
    else:run(args)


if __name__=='__main__':main()
