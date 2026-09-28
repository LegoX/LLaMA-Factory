import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[2]


def main():
    parser = argparse.ArgumentParser(description="CPU-only dependency/config checks; no model loading or training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
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
        "nvidia-cudnn-cu12": "9.26.0.51",
    }
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f"{name}: expected {version}, got {actual}")
    for name, directory in {
        "mcore_adapter": ROOT / "src/mcore_adapter",
        "llamafactory": ROOT / "src/llamafactory",
    }.items():
        spec = importlib.util.find_spec(name)
        if spec is None or Path(spec.origin).resolve().parent != directory:
            raise ValueError(f"{name} resolves outside this checkout: {spec.origin if spec else None}")
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
        tokenized = config.get("tokenized_path")
        if tokenized:
            if not Path(tokenized).is_dir():
                raise ValueError(f"tokenized_path does not exist: {tokenized}")
        else:
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
    print("This does not replace a multi-GPU optimizer-step acceptance run")


if __name__ == "__main__":
    main()
