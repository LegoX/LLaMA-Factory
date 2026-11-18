"""Single YAML, foreground, single-node LlamaFactory launcher."""

import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PROFILE_ENV = {
    "PROBE_CPU_OFFLOAD_LOAD": "1",
    "PROBE_CHUNK_CE": "1",
    "PROBE_ACTIVATION_OFFLOAD": "1",
    "PROBE_TEXT_ONLY": "1",
    "PROBE_GDN_SEGMENT": "16384",
    "PROBE_FP32_CE": "1",
    "PROBE_NESTED_CHECKPOINT": "1",
    "PROBE_MLP_CHUNK": "4096",
    "PROBE_RECOMPUTE_OFFLOAD": "1",
    "PROBE_NORM_CHUNK": "4096",
    "PROBE_VERIFY_WEIGHTS": "1",
    "PROBE_SAVE_MODEL": "1",
    "PROBE_CE_SYNC": "1",
    "PROBE_CE_EMPTY_CACHE": "1",
    "PROBE_ATTN_CHUNK": "0",
    "PROBE_CHECKPOINT_CPU_CACHE": "0",
    "PROBE_NORM_ELEMENT_BUDGET": "20971520",
    "PROBE_CHECKPOINT_GROUP": "2",
    "PROBE_GROUP_REENTRANT": "1",
    "PROBE_SHARD_LOSS_HEAD": "1",
    "PROBE_DISABLE_BACKWARD_PREFETCH": "1",
    "PROBE_LAYER_TRACE": "0",
}
OPTIONS = {
    "profile",
    "nproc_per_node",
    "device",
    "dataset_info",
    "cache_dir",
    "min_host_available_gib",
    "optimizations",
    "offline",
}
OPTIMIZATIONS = {
    "gdn_segment": "PROBE_GDN_SEGMENT",
    "mlp_chunk": "PROBE_MLP_CHUNK",
    "norm_chunk": "PROBE_NORM_CHUNK",
    "norm_element_budget": "PROBE_NORM_ELEMENT_BUDGET",
    "checkpoint_group": "PROBE_CHECKPOINT_GROUP",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_value(value, base):
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return str((p if p.is_absolute() else base / p).resolve())


def load_config(filename):
    filename = Path(filename).resolve()
    train = yaml.safe_load(filename.read_text())
    require(isinstance(train, dict), "YAML must be a mapping")
    train = dict(train)
    launch = train.pop("launcher", {})
    require(isinstance(launch, dict), "launcher must be a mapping")
    require(not (set(launch) - OPTIONS), f"Unknown launcher options: {set(launch) - OPTIONS}")
    launch = dict(launch)
    launch.setdefault("profile", "standard")
    launch.setdefault("device", "npu")
    launch.setdefault("nproc_per_node", 1)
    launch.setdefault("cache_dir", "/cache/llamafactory-training")
    launch.setdefault("min_host_available_gib", 128 if launch["profile"] == "qwen36_fsdp2" else 0)
    launch.setdefault("offline", True)
    require(launch["profile"] in ("standard", "qwen36_fsdp2"), "Unknown profile")
    require(launch["device"] in ("npu", "cuda"), "device must be npu or cuda")
    require(type(launch["nproc_per_node"]) is int and launch["nproc_per_node"] > 0, "nproc_per_node must be positive")
    require(float(launch["min_host_available_gib"]) >= 0, "Memory threshold cannot be negative")
    for key in ["model_name_or_path", "output_dir"]:
        require(key in train, f"Missing {key}")
    # Local models are intentional: reproducible offline jobs with explicit mounted inputs.
    for key in ["model_name_or_path", "output_dir", "dataset_dir", "tokenized_path", "resume_from_checkpoint"]:
        if train.get(key) is not None:
            require(isinstance(train[key], str), f"{key} must be an explicit path string")
            train[key] = path_value(train[key], filename.parent)
    launch["cache_dir"] = path_value(launch["cache_dir"], filename.parent)
    require(Path(train["model_name_or_path"], "config.json").is_file(), "Local model config.json missing")
    require(not train.get("overwrite_output_dir", False), "Use a new output_dir; automatic overwrite is disabled")
    require(train.get("do_train") is True, "This launcher requires do_train: true")
    info = launch.get("dataset_info")
    if info is not None:
        require("dataset_dir" not in train, "Use launcher.dataset_info OR dataset_dir, not both")
        require(isinstance(info, dict) and info, "dataset_info must be a nonempty mapping")
        info = json.loads(json.dumps(info))
        for entry in info.values():
            require("file_name" in entry, "Inline dataset_info requires local file_name")
            entry["file_name"] = path_value(entry["file_name"], filename.parent)
        launch["dataset_info"] = info
    else:
        require("dataset_dir" in train, "Provide launcher.dataset_info or dataset_dir")
        info = json.loads(Path(train["dataset_dir"], "dataset_info.json").read_text())
    selected = []
    for key in ["dataset", "eval_dataset"]:
        names = train.get(key, [])
        selected.extend([n.strip() for n in names.split(",")] if isinstance(names, str) else names)
    require(selected, "No dataset selected")
    data_files = []
    for name in selected:
        require(name in info, f"Dataset {name} missing in dataset_info")
        entry = info[name]
        require("file_name" in entry, "This portable launcher currently requires local dataset files")
        path = Path(entry["file_name"])
        if not path.is_absolute():
            path = Path(train["dataset_dir"]) / path
        require(path.is_file(), f"Dataset file missing: {path}")
        data_files.append(str(path.resolve()))
    profile = launch["profile"]
    if profile == "standard":
        require(not launch.get("optimizations"), "Qwen optimizations require qwen36_fsdp2 profile")
    else:
        model = json.loads(Path(train["model_name_or_path"], "config.json").read_text())
        require(launch["device"] == "npu", "qwen36_fsdp2 currently supports NPU only")
        require(
            model.get("model_type") == "qwen3_5" and model.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
            "qwen36_fsdp2 requires the supported Qwen3.5/3.6 dense architecture; use standard for other models",
        )
        require(
            not model.get("tie_word_embeddings", False)
            and not model.get("text_config", {}).get("tie_word_embeddings", False),
            "Qwen loss-head sharding requires untied weights",
        )
        require(
            train.get("stage", "sft") == "sft" and train.get("finetuning_type") == "full",
            "qwen36_fsdp2 requires full SFT",
        )
        require(
            train.get("bf16") is True and not train.get("fp16") and not train.get("pure_bf16"),
            "qwen36_fsdp2 requires BF16 compute / FP32 master weights",
        )
        require(train.get("per_device_train_batch_size", 1) == 1, "qwen36_fsdp2 requires per-device batch size 1")
        require(
            not train.get("packing")
            and not train.get("do_eval")
            and not train.get("do_predict")
            and not train.get("val_size", 0),
            "Optimized profile supports text-only, unpacked training without evaluation",
        )
        require(
            not train.get("deepspeed") and not train.get("fsdp") and not train.get("quantization_bit"),
            "Do not combine optimized FSDP2 with external sharding or quantization",
        )
        require(not train.get("streaming"), "Optimized sequential sampler requires a finite dataset")
        train.setdefault("gradient_accumulation_steps", 1)
        require(train["gradient_accumulation_steps"] > 0, "Invalid accumulation steps")
        opts = launch.get("optimizations", {})
        require(not (set(opts) - set(OPTIMIZATIONS)), "Unknown optimization setting")
        for name, value in opts.items():
            require(type(value) is int and value > 0, f"{name} must be a positive integer")
        group = opts.get("checkpoint_group", 2)
        require(
            model["text_config"]["num_hidden_layers"] % group == 0, "checkpoint_group must divide number of layers"
        )
    require(int(train.get("max_steps", -1)) != 0, "max_steps must be positive or -1")
    return filename, train, launch, data_files


def environment(launch):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROBE_", "TRAINING_"))}
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MALLOC_ARENA_MAX="2",
        MALLOC_TRIM_THRESHOLD_="134217728",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    env.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    env.setdefault("HCCL_EXEC_TIMEOUT", "1800")
    cache = launch["cache_dir"]
    env.update(
        TRITON_CACHE_DIR=str(Path(cache) / "triton"),
        HF_HOME=str(Path(cache) / "huggingface"),
        TORCH_EXTENSIONS_DIR=str(Path(cache) / "torch_extensions"),
        IMAGE_PLACEHOLDER="<|lf_media_image|>",
        VIDEO_PLACEHOLDER="<|lf_media_video|>",
        AUDIO_PLACEHOLDER="<|lf_media_audio|>",
    )
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1" if launch["offline"] else "0"
    if launch["device"] == "npu":
        env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    if launch["profile"] == "qwen36_fsdp2":
        env.update(PROFILE_ENV)
        for key, value in launch.get("optimizations", {}).items():
            env[OPTIMIZATIONS[key]] = str(value)
    return env


def preflight(train, launch):
    import torch

    if launch["device"] == "npu":
        import torch_npu  # noqa

        count = torch.npu.device_count()
    else:
        count = torch.cuda.device_count()
    require(
        count == launch["nproc_per_node"],
        f"YAML requests {launch['nproc_per_node']} devices, allocation exposes {count}; adjust YAML or allocation",
    )
    if launch["profile"] == "qwen36_fsdp2":
        for name, version in {
            "torch": "2.7.1+cpu",
            "torch-npu": "2.7.1.post4",
            "transformers": "5.8.0",
            "accelerate": "1.11.0",
            "llamafactory": "0.9.6.dev0",
            "triton-ascend": "3.2.1",
        }.items():
            require(importlib.metadata.version(name) == version, f"Optimized profile requires {name}=={version}")
    require(available_gib() >= float(launch["min_host_available_gib"]), "Available host memory below YAML threshold")
    checkpoint = train.get("resume_from_checkpoint")
    if checkpoint:
        require(Path(checkpoint, "trainer_state.json").is_file(), "Resume checkpoint trainer_state.json missing")
        if launch["profile"] == "qwen36_fsdp2":
            sys.path.insert(0, str(ROOT / "runtime"))
            from checkpoint_verify import verify_checkpoint

            verify_checkpoint(checkpoint, launch["nproc_per_node"])
    else:
        out = Path(train["output_dir"])
        require(
            not out.exists() or not any(out.iterdir()),
            "output_dir is nonempty; set an explicit resume_from_checkpoint or choose a new directory",
        )


def available_gib():
    return next(
        int(line.split()[1]) / 1024**2
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def accelerate_config(launch):
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "machine_rank": 0,
        "num_processes": launch["nproc_per_node"],
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
        "main_training_function": "main",
        "fsdp_config": {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_offload_params": True,
            "fsdp_reshard_after_forward": True,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3_5DecoderLayer,Qwen3_5VisionBlock,Embedding,ChunkedLossHead",
            "fsdp_version": 2,
        },
    }


def launch_job(filename, train, launch, data_files):
    cache = Path(launch["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    lock = (cache / "training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("A training job already holds this runtime lock")
    preflight(train, launch)
    out = Path(train["output_dir"])
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run = out / "runs" / run_id
    run.mkdir(parents=True)
    shutil.copy2(filename, run / "job.yaml")
    shutil.copytree(ROOT / "runtime", run / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    env = environment(launch)
    # Bind tokenized caches to task parameters, dataset bytes, and tokenizer/config bytes.
    fingerprint = {
        "train": {
            k: v for k, v in train.items() if k not in ("output_dir", "resume_from_checkpoint", "tokenized_path")
        },
        "profile": launch["profile"],
        "dataset_info": launch.get("dataset_info"),
        "files": {p: file_hash(p) for p in data_files},
    }
    model = Path(train["model_name_or_path"])
    fingerprint["tokenizer"] = {
        p.name: file_hash(p)
        for p in model.iterdir()
        if p.is_file() and (p.suffix in (".json", ".jinja", ".txt", ".model"))
    }
    if "dataset_dir" in train:
        fingerprint["dataset_info_sha256"] = file_hash(Path(train["dataset_dir"]) / "dataset_info.json")
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    tokenized = Path(train.get("tokenized_path", str(out / "tokenized" / digest[:20])))
    marker = tokenized.parent / (tokenized.name + ".task.json")
    if tokenized.exists() and any(tokenized.iterdir()):
        require(
            marker.exists() and json.loads(marker.read_text())["sha256"] == digest,
            "Existing tokenized cache has no matching task fingerprint; choose a new tokenized_path",
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"sha256": digest, "inputs": fingerprint}, indent=2))
    train["tokenized_path"] = str(tokenized)
    if launch.get("dataset_info"):
        (run / "dataset_info.json").write_text(json.dumps(launch["dataset_info"], indent=2))
        train["dataset_dir"] = str(run)
    (run / "train.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
    if launch["profile"] == "qwen36_fsdp2":
        acc = run / "accelerate.yaml"
        acc.write_text(yaml.safe_dump(accelerate_config(launch)))
        env["PROBE_ACCELERATE_CONFIG"] = str(acc)
        env["PROBE_CE_TRACE_DIR"] = str(out)
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(acc),
            str(run / "runtime/train.py"),
            str(run / "train.yaml"),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(launch["nproc_per_node"]),
            str(run / "runtime/standard_train.py"),
            str(run / "train.yaml"),
        ]
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "command": command,
                "source_sha256": {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()},
                "profile": launch["profile"],
                "optimizations": {k: v for k, v in env.items() if k.startswith(("PROBE_", "TRAINING_"))},
            },
            indent=2,
        )
    )
    cwd = cache / "runtime"
    cwd.mkdir(exist_ok=True)
    status = {"phase": "starting", "run_id": run_id, "command": command, "log": str(run / "train.log")}
    status_path = run / "status.json"

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    print(f"Run directory: {run}", flush=True)
    with (run / "train.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        status.update(phase="running", pid=process.pid)
        save()

        def tee():
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except BrokenPipeError:
                    pass

        thread = threading.Thread(target=tee, daemon=True)
        thread.start()
        stopping = [None]

        def stop(signum, frame):
            if process.poll() is None:
                stopping[0] = stopping[0] or time.monotonic()
                status["stop_reason"] = f"signal {signum}"
                os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while process.poll() is None:
            if available_gib() < float(launch["min_host_available_gib"]) and stopping[0] is None:
                stop(signal.SIGTERM, None)
                status["stop_reason"] = "Host memory below configured threshold"
                save()
            if stopping[0] is not None and time.monotonic() - stopping[0] > 30:
                os.killpg(process.pid, signal.SIGKILL)
            time.sleep(1)
        code = process.wait()
        thread.join(timeout=10)
    status.update(phase="finished" if code == 0 and stopping[0] is None else "failed", exit_code=code)
    save()
    return code if code else (1 if stopping[0] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate configuration and paths without touching devices or outputs"
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Also check device allocation, dependencies and checkpoint structure; do not train",
    )
    args = parser.parse_args()
    filename, train, launch, data_files = load_config(args.config)
    if args.dry_run or args.check:
        if args.check:
            preflight(train, launch)
        print(
            json.dumps(
                {
                    "mode": "check" if args.check else "dry-run",
                    "launcher": launch,
                    "training": train,
                    "data_files": data_files,
                },
                indent=2,
            )
        )
        return 0
    return launch_job(filename, train, launch, data_files)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
