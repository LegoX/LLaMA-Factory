import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

from prepare_adapter import BUNDLE, verify_adapter


def main():
    parser = argparse.ArgumentParser(description="CPU-only dependency/config checks; no model loading or training")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    verify_adapter(args.adapter)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["USE_MCA"] = "1"
    expected = {
        "torch": "2.10.0+cu128",
        "transformers": "5.6.0",
        "megatron-core": "0.18.2",
        "transformer-engine": "2.18.0",
        "flash-attn": "2.8.3.post1",
        "flash-linear-attention": "0.5.1",
        "fla-core": "0.5.1",
        "mcore-adapter": "0.10.0.dev0",
    }
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f"{name}: expected {version}, got {actual}")
    for name, directory in {
        "mcore_adapter": args.adapter.resolve() / "mcore_adapter/src/mcore_adapter",
        "llamafactory": BUNDLE.parents[2] / "src/llamafactory",
    }.items():
        spec = importlib.util.find_spec(name)
        if spec is None or Path(spec.origin).resolve().parent != directory:
            raise ValueError(f"{name} resolves outside the intended checkout")
    import yaml
    from mcore_adapter.models.qwen3_5.config_qwen3_5 import Qwen3_5Config
    from mcore_adapter.training_args import Seq2SeqTrainingArguments

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

    config = yaml.safe_load(args.config.read_text())
    fields = set()
    for cls in [Seq2SeqTrainingArguments, DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments]:
        fields.update(field.name for field in dataclasses.fields(cls))
    unknown = config.keys() - fields
    if unknown:
        raise ValueError(f"Unknown training fields: {sorted(unknown)}")
    model_fields = {field.name for field in dataclasses.fields(Qwen3_5Config)}
    unknown = config.get("additional_configs", {}).keys() - model_fields
    if unknown:
        raise ValueError(f"Unknown model fields: {sorted(unknown)}")
    if args.check_data:
        model = Path(config["model_name_or_path"])
        if not (model / "config.json").is_file():
            raise ValueError("Set model_name_or_path to an existing local HF checkpoint")
        data_dir = Path(config["dataset_dir"])
        info = json.loads((data_dir / "dataset_info.json").read_text())
        for name in config["dataset"].split(","):
            entry = info[name.strip()]
            if "file_name" in entry and not (data_dir / entry["file_name"]).is_file():
                raise ValueError(f"Missing dataset file for {name}")
        output = Path(config["output_dir"])
        if output.exists():
            raise ValueError("Choose a new output directory; preflight refuses existing run directories")
    print("CPU preflight passed: imports, versions, source identity, and all YAML/model fields")
    print("This does not replace an eight-GPU optimizer-step acceptance run")


if __name__ == "__main__":
    main()
