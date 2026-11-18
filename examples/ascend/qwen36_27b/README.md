# Qwen3.6-27B training on Ascend NPUs

A single-YAML, single-node training launcher with optional Qwen3.5/3.6 dense-model memory optimizations. It adds a standalone launcher and runtime patches; it does not modify the repository's LlamaFactory implementation.

## Validation and upstream relationship

This branch is based directly on upstream [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) commit `6d1059c727450a4f58e56ad204d01a71abad7781` (`0.9.6.dev0`), the exact framework source used by the validated environment. The framework source is included in this branch; install it from this checkout rather than installing a separate upstream revision. Its built-in Qwen NPU GDN adapter is upstream functionality. The additional patches here cover segmented GDN, chunked CE/MLP/norm, grouped activation recomputation, CPU offload, separate embedding/head sharding, FP32 gradient reduction and clipping, and FSDP2 optimizer-state restore. These patches also interact with Transformers, Accelerate and PyTorch internals; arbitrary dependency upgrades are not validated.

Validated environment:

| Component | Version |
|---|---|
| Python | 3.11.15, Linux aarch64 |
| CANN | 9.0.0 |
| PyTorch | 2.7.1+cpu with torch_npu |
| torch_npu | 2.7.1.post4 |
| Transformers / Tokenizers | 5.8.0 / 0.22.2 |
| Accelerate | 1.11.0 |
| Triton / Triton Ascend | 3.5.0 / 3.2.1 |
| Ascend driver tested | 25.5.1 and 25.5.5 |

The earlier task-specific launcher completed one real Qwen3.6-27B, 256K optimizer step on 16 NPUs and saved a structurally verified checkpoint before interruption. The portable launcher has completed two-NPU small-model training and full checkpoint resume; all 77 final weight tensors matched uninterrupted training bit for bit. A different-architecture tiny Llama model also completed two-NPU training through the standard profile. The portable launcher has **not** completed real 27B/256K/16-NPU resume or the full production training run. CPU offload for the original 16-NPU task used approximately 2 TiB host RAM; smaller device allocations are not a drop-in replacement.

## Environment preparation

Use a compatible Ascend base image providing CANN, PyTorch and torch_npu at the versions above. Do not replace the vendor stack with a generic CUDA/CPU environment. Clone this branch and install its included framework source with the pinned runtime requirements in an isolated image/environment:

```bash
git clone --branch qwen36-27b-npu-training https://github.com/LegoX/LLaMA-Factory.git
cd LLaMA-Factory
python -m pip install . -r examples/ascend/qwen36_27b/requirements.txt
# Installation order matters: the Ascend package overlays Triton backend files.
python -m pip install triton==3.5.0
python -m pip install --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple triton-ascend==3.2.1
```

This is not an instruction to upgrade or reinstall dependencies in a running training job. `--check` verifies the optimized profile's pinned core versions and device allocation. The standard profile delegates model support to the installed LlamaFactory version.

## Configure and launch

Copy `configs/qwen36-256k.yaml` and replace its model, dataset and output paths. All top-level keys except `launcher` are standard LlamaFactory options. The extra `launcher` mapping selects the execution profile, device count, local cache location and optional inline dataset definitions.

```bash
bash examples/ascend/qwen36_27b/train.sh /path/to/job.yaml --dry-run
bash examples/ascend/qwen36_27b/train.sh /path/to/job.yaml --check
bash examples/ascend/qwen36_27b/train.sh /path/to/job.yaml
```

`--dry-run` validates configuration and local paths without using devices or writing outputs. `--check` additionally checks allocated devices, dependencies and resume checkpoint structure. A regular launch stays in the foreground, forwards termination signals to the worker group and returns its exit status. Do not add `nohup` or `&` in a managed training job.

Example launcher section:

```yaml
launcher:
  profile: qwen36_fsdp2
  device: npu
  nproc_per_node: 16
  min_host_available_gib: 128
  cache_dir: /cache/llamafactory-training
  dataset_info:
    train:
      file_name: /path/to/data/train.json
      formatting: sharegpt
      columns:
        messages: messages
      tags:
        role_tag: role
        content_tag: content
        user_tag: user
        assistant_tag: assistant
        system_tag: system
```

Alternatively, omit `launcher.dataset_info` and set the normal `dataset_dir` pointing to a directory containing `dataset_info.json`. Do not specify both. Currently this wrapper requires local models and dataset files; it does not download model/data repositories. Relative paths are resolved against the job YAML directory, with environment-variable expansion supported. Allocate exactly the configured visible device count on **one node**; multi-node launch is not supported.

## Profiles and task changes

- `standard`: native LlamaFactory via torchrun, without this example's Qwen runtime patches. See `configs/standard-lora.yaml`. Change model, template, data, finetuning type and hyperparameters as supported by LlamaFactory. NPU execution is tested; the CUDA option is not tested in this environment. The sample config is not a model-capacity guarantee.
- `qwen36_fsdp2`: the memory-optimized text-only full-SFT path for the `Qwen3_5ForConditionalGeneration` dense architecture. It requires BF16 compute, untied embeddings, per-device batch size 1, finite unpacked datasets, and no evaluation/prediction. It does not support LoRA, quantization, actual multimodal samples, MoE models, external FSDP configuration or DeepSpeed. Vision/projector freezing follows LlamaFactory defaults. Other model sizes sharing this architecture still require validation.

The optimized profile uses sequential sampling padded to `device_count × gradient_accumulation_steps`, sample-mean loss and the recovered weight-decay grouping. The standard profile retains LlamaFactory's sampler behavior. There is no hidden 48-step limit: training duration comes from the YAML's epochs/max_steps. Historical sample hashes and private experiment data are not included.

Optional positive-integer `launcher.optimizations` settings: `gdn_segment`, `mlp_chunk`, `norm_chunk`, `norm_element_budget`, `checkpoint_group`. Defaults match the earlier 256K task. The checkpoint group must divide the model layer count. Revalidate memory usage and numerical behavior after changing these settings.

Changing model/data should normally create a new output directory and omit `resume_from_checkpoint`. To continue the same task, set an explicit checkpoint directory:

```yaml
resume_from_checkpoint: /path/to/outputs/task/checkpoint-1
```

Existing nonempty output directories are rejected unless resume is explicitly configured. Automatic overwrite is disabled. For optimized checkpoints, the launcher checks DCP metadata, referenced shard byte ranges, scheduler state and per-rank RNG files before starting. Structural verification is not a substitute for a full restore test.

## Outputs, cache and image installation

Each run saves YAML/configuration and runtime source snapshots, a hash manifest, log and exit status under `output_dir/runs/<UTC timestamp>/`. Logs are also streamed to the platform console. Model/optimizer checkpoints remain under `output_dir`.

Tokenized caches default to `output_dir/tokenized/<fingerprint>/`, keyed by task parameters, dataset bytes, model configuration and tokenizer files. Existing explicitly configured caches must carry a matching fingerprint created by this launcher. Reusing unverified caches from another launcher is rejected.

The default `/cache/llamafactory-training` holds rebuildable compilation caches and the worker's local working directory. Use a local filesystem here: compiler startup from an s3fs working directory previously stalled. Keep outputs on writable persistent storage. The optimized launcher defaults to stopping training when host available memory falls below 128 GiB; adjust `launcher.min_host_available_gib` only for an appropriate workload.

To install the required code into the image's system filesystem:

```bash
bash examples/ascend/qwen36_27b/install.sh
# Or choose another writable system location:
bash examples/ascend/qwen36_27b/install.sh /your/system/path
```

The default destination is `/usr/local/share/llamafactory-training`. Save a new platform image after installation; mounted models, data and outputs are not assumed to be included. A platform code directory can contain only the task YAML:

```bash
bash /usr/local/share/llamafactory-training/train.sh "${MA_JOB_DIR}/my-job/job.yaml"
```

## Tests

Run CPU-only configuration tests:

```bash
python examples/ascend/qwen36_27b/tests/test_config.py
```

The following optional end-to-end checks use **two idle NPUs**, local scratch space and the local Qwen3.6 tokenizer/processor files. They create randomly initialized small models and synthetic data; no production weights or user dataset are used. Use an empty work directory and run jobs sequentially:

```bash
python examples/ascend/qwen36_27b/tests/prepare_tiny.py --work-dir /cache/qwen-smoke --tokenizer-source /path/to/models/Qwen3.6-27B
bash examples/ascend/qwen36_27b/train.sh /cache/qwen-smoke/train.yaml
bash examples/ascend/qwen36_27b/train.sh /cache/qwen-smoke/resume.yaml
python examples/ascend/qwen36_27b/tests/compare_resume.py --work-dir /cache/qwen-smoke
python examples/ascend/qwen36_27b/tests/prepare_standard.py --work-dir /cache/qwen-smoke
bash examples/ascend/qwen36_27b/train.sh /cache/qwen-smoke/standard.yaml
```

Generated datasets, model files, logs, checkpoints and reports belong outside the repository. Existing small-model tests do not imply full-scale numerical equivalence with GPU/Megatron training.

## License

Apache-2.0, consistent with the surrounding LlamaFactory repository. Retain upstream notices in derived files. Model and dataset licenses are separate and their assets are not distributed here.
