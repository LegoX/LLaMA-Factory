"""Check the packaged interpreter, imports and image-owned runtime paths."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

profile = sys.argv[1]
expected = {'torch': '2.10.0+cpu', 'torch-npu': '2.10.0'}
expected.update({'transformers': '5.8.0', 'llamafactory': '0.9.6.dev0'})
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    if actual != version:
        raise RuntimeError(f'{name}: expected {version}, found {actual}')
modules = ['torch', 'torch_npu', 'llamafactory', 'transformers', 'accelerate', 'datasets', 'peft', 'trl', 'safetensors', 'triton']
if True:
    modules.append('hyper_parallel')
paths = {}
for name in modules:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to('/usr/local'):
        raise RuntimeError(f'{name} depends on a non-image runtime path: {path}')
    paths[name] = str(path)
if True:
    from hyper_parallel.integration.llamafactory import HyperParallelArguments
    HyperParallelArguments.from_dict({'enable_offload': True})
from llamafactory.train.tuner import run_exp
for module in list(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if filename and Path(filename).is_absolute() and str(Path(filename).resolve()).startswith(('/mnt/', '/cache/', '/home/ma-user/work/')):
        raise RuntimeError(f'Imported code depends on mounted storage: {filename}')
print(json.dumps({'passed': True, 'profile': profile, 'python': sys.executable, 'versions': expected, 'module_paths': paths}, indent=2))
