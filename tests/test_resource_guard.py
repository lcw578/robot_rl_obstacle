"""实际验证子进程资源限制、低内存不启动、超时结束进程组；不加载整车模型。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from resource_guard import guarded_run,available_memory_mb


class ResourceGuardTests(unittest.TestCase):
    def test_explicit_cpu_ids_are_passed_to_child(self):
        if available_memory_mb()<1600:self.skipTest('系统余量不足，不启动测试子进程')
        allowed=sorted(os.sched_getaffinity(0));chosen=[allowed[-1]]
        script='import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'
        with tempfile.TemporaryFile(mode='w+') as log:
            result=guarded_run([sys.executable,'-c',script],log,Path.cwd(),os.environ.copy(),
                timeout=5,memory_mb=512,reserve_mb=1024,cpu_ids=chosen)
            log.seek(0);actual=json.loads(log.read())
        self.assertEqual(result['exit_code'],0)
        self.assertEqual(actual,chosen);self.assertEqual(result['cpu_ids'],chosen)

    def test_invalid_explicit_cpu_ids_do_not_start_process(self):
        invalid=max(os.sched_getaffinity(0))+1
        with patch('resource_guard.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):guarded_run([],None,Path.cwd(),{},cpu_ids=[invalid])
            with self.assertRaises(ValueError):guarded_run([],None,Path.cwd(),{},cpu_ids=[])
            spawn.assert_not_called()

    def test_hard_limit_affinity_threads_and_priority_are_inherited(self):
        if available_memory_mb()<1600:self.skipTest('系统余量不足，不启动测试子进程')
        script='import json,resource,os; print(json.dumps(dict(memory=resource.getrlimit(resource.RLIMIT_AS)[0],cores=len(os.sched_getaffinity(0)),threads=os.environ["OPENBLAS_NUM_THREADS"],nice=os.nice(0))))'
        with tempfile.TemporaryFile(mode='w+') as log:
            result=guarded_run([sys.executable,'-c',script],log,Path.cwd(),os.environ.copy(),timeout=5,memory_mb=512,reserve_mb=1024,cores=1)
            log.seek(0);values=json.loads(log.read())
        self.assertEqual(result['exit_code'],0);self.assertEqual(values['memory'],512*1024*1024)
        self.assertEqual(values['cores'],1);self.assertEqual(values['threads'],'1');self.assertGreaterEqual(values['nice'],10)

    def test_low_memory_does_not_start_command(self):
        with patch('resource_guard.available_memory_mb',return_value=128),tempfile.TemporaryFile() as log:
            result=guarded_run(['/nonexistent-command-must-not-be-started'],log,Path.cwd(),os.environ.copy(),memory_mb=512,reserve_mb=1024)
        self.assertEqual(result['resource_status'],'not_started_low_memory');self.assertIsNone(result['exit_code'])

    def test_timeout_cleans_child_process_group(self):
        if available_memory_mb()<1600:self.skipTest('系统余量不足，不启动测试子进程')
        script='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(15)"]); print(p.pid,flush=True);time.sleep(15)'
        with tempfile.TemporaryFile(mode='w+') as log:
            result=guarded_run([sys.executable,'-c',script],log,Path.cwd(),os.environ.copy(),timeout=.5,memory_mb=512,reserve_mb=1024,cores=1)
            log.seek(0);child=int(log.read().strip())
        self.assertEqual(result['resource_status'],'case_timeout')
        status=Path(f'/proc/{child}/stat')
        if status.exists():self.assertIn(status.read_text().rsplit(') ',1)[1].split()[0],('Z','X'))


if __name__=='__main__':unittest.main()
