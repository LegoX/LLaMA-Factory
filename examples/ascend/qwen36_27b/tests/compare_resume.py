import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
root = args.work_dir.resolve()


def load(path):
    reader = dcp.FileSystemReader(path)
    meta = reader.read_metadata()
    state = {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype)
        for key, value in meta.state_dict_metadata.items()
    }
    dcp.load(state, storage_reader=reader)
    return state


expected = load(root / "full/checkpoint-2/pytorch_model_fsdp_0")
actual = load(root / "resumed/checkpoint-2/pytorch_model_fsdp_0")
assert expected.keys() == actual.keys()
different = [name for name in expected if not torch.equal(expected[name], actual[name])]
max_abs = max(float((expected[n].float() - actual[n].float()).abs().max()) for n in expected if expected[n].numel())
result = {
    "tensor_count": len(expected),
    "different_tensors": different,
    "max_absolute_difference": max_abs,
    "parameters_bitwise_equal": not different,
}
(root / "end-to-end-resume-comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
assert not different, result
