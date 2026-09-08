import ast
import functools
import hashlib
import json
import math
import os
import unittest
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "examples/megatron/512k"


def load_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class BundleTests(unittest.TestCase):
    def test_checksums_and_scope(self):
        manifest = json.loads((BUNDLE / "manifest.json").read_text())
        assert len(manifest["files"]) == 6
        assert hashlib.sha256((BUNDLE / "mcore-adapter.patch").read_bytes()).hexdigest() == manifest["patch_sha256"]
        for name, expected in manifest["llamafactory_files"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name in manifest["files"]:
            assert name.startswith("mcore_adapter/src/mcore_adapter/")
            assert ".." not in Path(name).parts

    def test_loss_weights_align_with_shifted_labels(self):
        namespace = {"functools": functools, "Sequence": Sequence, "Any": Any, "deepcopy": deepcopy}
        wrapper = load_definition(ROOT / "src/llamafactory/train/mca/workflow.py", "_data_collator_wrapper", namespace)
        collator = wrapper(lambda features: {"features": features})
        actual = collator(
            [
                {
                    "input_ids": [1, 2, 3],
                    "labels": [-100, 2, 3],
                    "loss_weights": [0.0, 0.5, 1.0],
                    "attention_mask": [1, 1, 1],
                }
            ]
        )["features"][0]
        assert actual["input_ids"] == [1, 2]
        assert actual["labels"] == [2, 3]
        assert actual["loss_weights"] == [0.5, 1.0]
        assert actual["attention_mask"] == [1, 1]

    def test_variable_length_padding_condition(self):
        tree = ast.parse((ROOT / "src/llamafactory/train/mca/workflow.py").read_text())
        function = next(node for node in tree.body if getattr(node, "name", None) == "run_sft")
        expression = next(
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pad_to_max" for target in node.targets)
        )
        from types import SimpleNamespace

        for experts, variable, expected in [
            (4, True, False),
            (4, False, True),
            (1, False, False),
            (None, True, False),
        ]:
            args = SimpleNamespace(expert_model_parallel_size=experts, variable_seq_lengths=variable)
            assert eval(compile(ast.Expression(expression), "padding", "eval"), {"training_args": args}) == expected

    def test_sampler_epochs_cover_all_samples_and_rotate_padding(self):
        adapter = os.environ.get("MCA_512K_ADAPTER")
        if not adapter:
            self.skipTest("Set MCA_512K_ADAPTER to the patched checkout to test its sampler")
        from torch.utils.data import Dataset, SequentialSampler

        sampler_type = load_definition(
            Path(adapter) / "mcore_adapter/src/mcore_adapter/trainer/trainer.py",
            "PaddedSequentialSampler",
            {"SequentialSampler": SequentialSampler, "Dataset": Dataset, "math": math},
        )
        sampler = sampler_type(list(range(2481)), 64)
        assert len(sampler) == 2496
        for epoch in range(3):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            assert indices[:2481] == list(range(2481))
            assert indices[2481:] == list(range(epoch * 15, (epoch + 1) * 15))
        assert len(sampler_type([], 64)) == 0
        assert list(sampler_type([], 64)) == []
        assert list(sampler_type([0, 1], 2)) == [0, 1]
        with self.assertRaises(ValueError):
            sampler_type([0], 0)


if __name__ == "__main__":
    unittest.main()
