"""轻量离线任务守护：串行任务的CPU/地址空间限制、总RSS/系统余量监视。

不导入MuJoCo/NumPy。资源不足是计算未完成，不能标成机器人规划失败。
Linux专用；prlimit硬限制每进程地址空间，总RSS监视按进程组保守累加。
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def available_memory_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024
    raise RuntimeError('无法读取MemAvailable，不启动资源密集任务')


def process_rss_mb():
    """当前训练进程的RSS；用于趋势诊断，不替代进程组资源守护。"""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    raise RuntimeError('无法读取当前进程RSS')


def group_rss_mb(pgid):
    rss=0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            stat=(entry/'stat').read_text().rsplit(') ',1)[1].split()
            if int(stat[2])!=pgid:continue
            for line in (entry/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):rss+=int(line.split()[1]);break
        except (OSError,ValueError,IndexError):continue
    return rss/1024


def startup_memory_message(result):
    """启动被拒绝时解释预算，不把资源不足说成测试/IK/规划失败。"""
    if result.get('resource_status')!='not_started_low_memory':return ''
    available=result['available_before_mb'];required=result.get('required_available_mb')
    if required is None:
        return f'内存不足，子进程未启动；当时可用{available:.0f} MiB。'
    return (f'内存不足，子进程未启动：可用{available:.0f} MiB；'
            f'需要至少{required:.0f} MiB（任务预算{result["memory_budget_mb"]:.0f}'
            f' + 桌面预留{result["memory_reserve_mb"]:.0f}），'
            f'还差约{max(0.,required-available):.0f} MiB。'
            '请先正常保存退出旧训练或关闭暂不用的应用，再重试；这不是测试或路径规划失败。')


def stop_group(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
    # 直接父进程先退出时仍可能存在孙进程，最终清理相同进程组。
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    # SIGKILL发出与内核实际完成退出之间有时序差；确认孙进程也不再运行后再开下一例。
    until=time.monotonic()+2
    while True:
        active=False
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                stat=(entry/'stat').read_text().rsplit(') ',1)[1].split()
                if int(stat[2])==process.pid and stat[0] not in ('Z','X'):active=True;break
            except (OSError,ValueError,IndexError):continue
        if not active:return True
        if time.monotonic()>=until:return False
        time.sleep(.02)


def guarded_run(command,log,cwd,environment,timeout=120,memory_mb=3072,reserve_mb=3072,cores=2,
                address_space_mb=None,cpu_ids=None,stop_file=None):
    allowed=os.sched_getaffinity(0)
    if cpu_ids is not None:
        selected=list(cpu_ids)
        if not selected or len(set(selected))!=len(selected) or not set(selected)<=allowed:
            raise ValueError('cpu_ids须为当前亲和范围内的不重复CPU编号')
    else:
        if type(cores) is not int or cores<1:raise ValueError('cores须为正整数')
        selected=sorted(allowed)[:cores]
    initial_available=available_memory_mb()
    if initial_available<reserve_mb+memory_mb:
        return {'exit_code':None,'resource_status':'not_started_low_memory','available_before_mb':initial_available,
            'memory_budget_mb':memory_mb,'memory_reserve_mb':reserve_mb,
            'required_available_mb':reserve_mb+memory_mb,
            'peak_group_rss_mb':0.,'elapsed_s':0.}
    for tool in ('prlimit','nice','taskset'):
        if shutil.which(tool) is None:raise RuntimeError('缺少资源限制工具:'+tool)
    # Torch共享库可能映射较大虚拟空间；RSS守护仍按memory_mb限制，不把映射量当实际RAM。
    address_limit=memory_mb if address_space_mb is None else address_space_mb
    # CUDA驱动会保留大段虚拟地址；此模式沿用系统地址空间限制，物理内存监视照常。
    prefix=[] if address_limit=='inherit' else ['prlimit',f'--as={int(address_limit)*1024*1024}','--']
    wrapped=[*prefix,'nice','-n','10','taskset','-c',','.join(str(c) for c in selected),*command]
    env=environment.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    start=time.monotonic();peak=0.;minimum=initial_available;status='completed'
    process=subprocess.Popen(wrapped,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # 仅训练调用显式启用：终端Ctrl+C先请求子进程正常保存，而非立即SIGTERM。
    # 其他离线工具保持原中断行为；资源越限仍优先强制结束，不撤销桌面保护。
    old_interrupt=None
    if stop_file is not None:
        stop_file=Path(stop_file)
        def request_save(signum,frame):
            stop_file.touch()
            print('已请求停止训练并保存模型/经验池，请等待终端返回；不要关闭终端。',flush=True)
        old_interrupt=signal.signal(signal.SIGINT,request_save)
    try:
        while process.poll() is None:
            memory=group_rss_mb(process.pid);available=available_memory_mb()
            peak=max(peak,memory);minimum=min(minimum,available)
            if memory>memory_mb:status='stopped_group_memory_limit'
            elif available<reserve_mb:status='stopped_low_system_memory'
            elif time.monotonic()-start>timeout:status='case_timeout'
            if status!='completed':stop_group(process);break
            time.sleep(.25)
    except BaseException:
        stop_group(process);raise
    finally:
        if old_interrupt is not None:signal.signal(signal.SIGINT,old_interrupt)
    # 若直接子进程异常结束而孙进程尚未退出，不留下脱离监控的后台任务。
    if not stop_group(process):status='stopped_cleanup_incomplete'
    return {'exit_code':process.returncode,'resource_status':status,'available_before_mb':initial_available,
        'minimum_available_mb':minimum,'peak_group_rss_mb':peak,'elapsed_s':time.monotonic()-start,
        'cpu_ids':selected,
        'manual_stop_requested':bool(stop_file is not None and stop_file.exists())}
