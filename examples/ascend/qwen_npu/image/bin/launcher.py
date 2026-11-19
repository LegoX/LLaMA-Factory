"""Single-YAML launcher for the image-owned HyperParallel environment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import yaml

ROOT = Path(__file__).resolve().parent.parent


def prepare(filename, accelerate=None):
    filename = Path(filename).resolve()
    train = yaml.safe_load(filename.read_text())
    if not isinstance(train, dict):
        raise ValueError('Task YAML must be a mapping')
    launch = train.pop('launcher', {})
    allowed = {'nproc_per_node', 'accelerate_config', 'min_host_available_gib', 'max_host_used_fraction'}
    if not isinstance(launch, dict) or set(launch) - allowed:
        raise ValueError('launcher supports only: ' + ', '.join(sorted(allowed)) +
                         '; old qwen36_fsdp2 profiles require migration to the new templates')
    def path(value):
        p = Path(os.path.expandvars(os.path.expanduser(str(value))))
        return str((p if p.is_absolute() else filename.parent / p).resolve())
    for key in ('model_name_or_path', 'dataset_dir', 'tokenized_path', 'output_dir',
                'resume_from_checkpoint', 'hyper_parallel_args'):
        if train.get(key) is not None:
            if not isinstance(train[key], str):
                raise ValueError(f'{key} must be an explicit path string')
            train[key] = path(train[key])
    if not train.get('model_name_or_path') or not Path(train['model_name_or_path'], 'config.json').is_file():
        raise ValueError('Local model config.json is missing')
    if not train.get('output_dir'):
        raise ValueError('output_dir is required')
    if train.get('do_train') is not True or train.get('use_hyper_parallel') is not True:
        raise ValueError('This entry requires do_train: true and use_hyper_parallel: true')
    output = Path(train['output_dir'])
    if train.get('overwrite_output_dir'):
        raise ValueError('Use a new output_dir instead of overwrite_output_dir')
    resume = train.get('resume_from_checkpoint')
    if resume and not Path(resume).is_dir():
        raise ValueError('resume_from_checkpoint directory is missing')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Nonempty output_dir requires an explicit resume_from_checkpoint')
    cache = train.get('tokenized_path')
    if not (cache and Path(cache, 'dataset_dict.json').is_file()):
        if not train.get('dataset_dir') or not Path(train['dataset_dir'], 'dataset_info.json').is_file():
            raise ValueError('Provide dataset_dir/dataset_info.json or an existing tokenized_path')
    hp = train.get('hyper_parallel_args')
    if not hp or not Path(hp).is_file():
        raise ValueError('hyper_parallel_args JSON is missing')
    explicit = accelerate or launch.get('accelerate_config')
    count = launch.get('nproc_per_node', 2)
    if type(count) is not int or count < 1:
        raise ValueError('nproc_per_node must be a positive integer')
    dist = yaml.safe_load(Path(path(explicit) if explicit else ROOT / 'configs/accelerate-16npu.yaml').read_text())
    if explicit:
        if 'nproc_per_node' in launch and dist['num_processes'] != count:
            raise ValueError('nproc_per_node conflicts with Accelerate num_processes')
        count = dist['num_processes']
    else:
        dist['num_processes'] = count
    if dist.get('num_machines') != 1 or dist.get('distributed_type') != 'FSDP':
        raise ValueError('This entry currently supports single-node FSDP only')
    cp = train.get('hyper_parallel_cp_size', 1)
    if type(cp) is not int or cp < 1 or count % cp:
        raise ValueError('NPU count must be divisible by hyper_parallel_cp_size')
    floor = float(launch.get('min_host_available_gib', 128))
    ceiling = float(launch.get('max_host_used_fraction', .60))
    if floor < 0 or not 0 < ceiling <= 1:
        raise ValueError('Invalid host-memory limits')
    return train, dist, {'min_host_available_gib': floor, 'max_host_used_fraction': ceiling}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('accelerate', nargs='?')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    train, dist, limits = prepare(args.task, args.accelerate)
    if not args.dry_run:
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.device_count() != dist['num_processes']:
            raise ValueError('Visible NPU count differs from requested nproc_per_node')
    if args.dry_run or args.check:
        print(json.dumps({'training': train, 'distributed': dist, 'memory_limits': limits}, indent=2))
        return 0
    memory = psutil.virtual_memory()
    if memory.available < limits['min_host_available_gib'] * 2**30:
        raise RuntimeError('Insufficient available host memory before launch')
    run = Path(train['output_dir']) / 'runs' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'task.yaml').write_text(yaml.safe_dump(train))
    (run / 'accelerate.yaml').write_text(yaml.safe_dump(dist))
    command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
               str(run / 'accelerate.yaml'), str(ROOT / 'bin/train.py'), str(run / 'task.yaml')]
    process = subprocess.Popen(command, start_new_session=True, cwd=str(Path(os.environ.get("NPU_TRAINING_SCRATCH", "/tmp/npu-training")) / "runtime"))
    reason = None
    def stop(signum=None, frame=None):
        nonlocal reason
        reason = reason or f'signal_{signum}'
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while process.poll() is None:
            memory = psutil.virtual_memory()
            usage, capacity = memory.total - memory.available, memory.total
            for current, maximum in [('/sys/fs/cgroup/memory/memory.usage_in_bytes', '/sys/fs/cgroup/memory/memory.limit_in_bytes'), ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max')]:
                if Path(current).exists() and Path(maximum).read_text().strip().isdigit():
                    usage = int(Path(current).read_text())
                    capacity = min(capacity, int(Path(maximum).read_text()))
                    break
            (run / 'resource-latest.json').write_text(json.dumps({'time': time.time(), 'used_bytes': usage, 'capacity_bytes': capacity, 'available_bytes': memory.available}))
            if memory.available < limits['min_host_available_gib'] * 2**30 or usage > capacity * limits['max_host_used_fraction']:
                reason = 'host_memory_limit'
                stop()
            else:
                time.sleep(5)
    finally:
        if process.poll() is None:
            stop()
        (run / 'exit.json').write_text(json.dumps({'returncode': process.returncode, 'stop_reason': reason}))
    return process.returncode if process.returncode else (1 if reason else 0)


if __name__ == '__main__':
    sys.exit(main())
