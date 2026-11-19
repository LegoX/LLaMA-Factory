# Long-context SFT integration

This integration extends `LegoX/LLaMA-Factory` `conghao/feature` at
`63455eb8346f88df211dc6ab6f79e668ac11fcc5` with the long-context training,
scoring and export features described below. Their implementation is maintained in this repository.
The adapter lives in `src/mcore_adapter`; there is no external adapter patch or
second training launcher. Private datasets, run histories, credentials and
machine-specific launch defaults are not part of this integration.

## Launch and environment

Use the pinned environment described in [the 512K guide](../examples/megatron/README_512k.md).
In particular, cuDNN fused attention with packed THD layout needs the branch's
cuDNN 9.26 environment. Preflight also accepts torch's cuDNN `9.10.2.21` with
explicit `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0`, which disables cuDNN attention.
The other pinned versions still apply; CPU tests do not certify GPU execution.

The common entry supports YAML overrides, single-node and multi-node runs:

```bash
# Topology and merged configuration only; does not import the MCA training stack.
DRY_RUN=1 bash examples/megatron/512k/run.sh /path/to/env /path/to/train.yaml learning_rate=4e-5

# Check installed versions, source identity, config fields and local input files.
PREFLIGHT_ONLY=1 bash examples/megatron/512k/run.sh /path/to/env /path/to/train.yaml

# Actual training. For multiple nodes export NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT on each node.
LF_MCA_PROFILE=1 bash examples/megatron/512k/run.sh /path/to/env /path/to/train.yaml
```

Single-node runs reset inherited distributed rank/rendezvous settings and bind
localhost. `CUDA_VISIBLE_DEVICES` remains under the caller's control. The effective
batch is `micro_batch × gradient_accumulation_steps × DP`, where
`DP = world_size / (TP × PP × CP)`; EP is not another independent multiplier.
Preflight uses the inherited HF training defaults for omitted microbatch and
accumulation settings. Parallel sizes follow the adapter's precedence: explicit
arguments, then `additional_configs`, then native checkpoint settings, then defaults.
Both `neat_packing` and MCA's `sequence_packing` enable
the CP packing alignment check.
Preflight, launcher and training workers resolve YAML/JSON interpolations after
applying command-line overrides, so they use the same effective paths and flags.

Use `source examples/megatron/512k/activate_shared.sh /path/to/env` for an interactive
shell. It works with a conda prefix or a virtualenv and selects this checkout.
It does not install packages or modify an existing environment.

Sanitized examples:

- [Qwen3.5 512K YaRN](../examples/megatron/qwen3_5_35b_a3b_base_512k_yarn.yaml):
  TP2/PP2/CP2/EP4, precision-aware optimizer and chunked vocabulary-parallel CE.
- [Qwen3.6 256K packed](../examples/megatron/qwen3_6_27b_256k_4node.yaml):
  four nodes, CP packing and seeded epoch shuffle.
- [Qwen3.8 256K supervision](../examples/megatron/qwen3_8_27b_256k.yaml):
  eight GPUs, TP4/PP2/CP1, turn masks and trajectory weights.

These are configurations, not GPU memory-fit or model-quality guarantees. For CP
packing set `LF_PACK_SUBSEQ_ALIGN` to a multiple of `2 × CP` before tokenization
and training. Rebuild tokenized caches after changing the template, masks,
weights, cutoff, alignment or any other preprocessing setting.

## Template and supervision

`template: qwen3_8` supports `reasoning_effort: xhigh` (default), `medium`, and
`low`. In SFT, historical thinking is preserved by default. Removing it requires
both `mask_history: true` and `preserve_thinking: false`; only the last assistant
turn is then supervised. With `mask_history: false`, earlier supervised turns
keep their thinking even if `preserve_thinking: false`. In single-turn inference,
`preserve_thinking: false` removes historical thinking independently. History ends
at the latest user query; reasoning in its subsequent tool-call chain is retained.
Tool responses do not start a new query. The template
retains the OpenAI function wrapper and the tokenizer's expected separators,
including when `tool_format: qwen3_8` is explicitly selected.

`turn_mask: true` enables per-message `turn_mask: true` on assistant/function
turns. Masked tokens stay in the input context but receive label `-100`; EOS
labels closing a masked turn are masked too. This composes with `mask_history`.
`drop_long_sequences: true` discards samples whose *complete* encoded length
exceeds `cutoff_len`, in both packed and unpacked SFT. The default remains truncation.

Register a trajectory-weighted ShareGPT dataset in its `dataset_info.json`:

```json
{
  "weighted_sft": {
    "file_name": "train.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "loss_weight": "loss_weight"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
  }
}
```

Example row:

```json
{
  "loss_weight": 0.5,
  "messages": [
    {"role": "user", "content": "Inspect the failing test."},
    {"role": "assistant", "content": "An incorrect attempt.", "turn_mask": true},
    {"role": "user", "content": "Try again."},
    {"role": "assistant", "content": "The corrected approach.", "loss_weight": 0.25}
  ]
}
```

Enable `trajectory_loss_weight: true`. Row weights must be finite numbers in
`(0, 1]`, explicitly mapped in the dataset registration. The feature currently
supports ordinary, unpacked MCA ShareGPT SFT. It rejects ranking/KTO and other
backends instead of silently dropping weights. Message-level `loss_weight`
continues to work in ShareGPT, including the HF SFT backend; zero masks that turn.
MCA trajectory weighting requires MTP to be disabled in the resolved model
configuration. Set `mtp_num_layers: 0` to override both checkpoint defaults and
`additional_configs`; validation runs before model allocation. Non-unit message
weights are also rejected when MTP is enabled: the MTP auxiliary objective does
not implement these weights.

**Loss normalization changes from the feature branch:**

```text
CE loss = sum(valid_token × turn_weight × trajectory_weight × token_CE)
          / count(valid_token)
```

A masked token contributes neither numerator nor denominator. Without a row or
turn weight its multiplier is 1. This deliberately replaces normalization by
`sum(weights)`: a uniform weight of 0.5 scales the CE gradient by 0.5, including
microbatch=1, and fractional weights cannot be truncated into integer token
counts. HF fractional weighting needs token logits; incompatible fused HF loss
paths fail explicitly. MCA's fused token CE remains supported. The loss callback
applies weights only at the final PP stage; CP compensation exists once in the
adapter and does not alter reported loss/token metrics.

These weights apply only to token CE. Megatron MoE auxiliary losses keep their
existing scaling, so the total gradient need not halve when a weight is 0.5.

In MCA, `calculate_per_token_loss: true` counts valid tokens across the
accumulated global batch before normalizing gradients. Its default, `false`,
normalizes each microbatch separately and then averages across microbatches and
data-parallel ranks. These objectives differ when valid lengths differ. An
all-masked microbatch has zero CE loss and zero valid tokens; it does not add a
dummy token to the global denominator.

HF weighted SFT uses the microbatch-normalized objective: it divides each
weighted token sum by that microbatch's valid-token count, then averages over
the actual accumulation window (including a partial final window). It does not
use the accumulated global-token denominator. In MCA per-token mode, logged
and evaluation losses use the reduced loss sum divided by the reduced token
count; training loss is independent of the logging interval.

## Pre-update scoring

```yaml
pretrain_score: true
pretrain_score_only: true       # omit/false to train after scoring
pretrain_score_microbatches: 16 # 1 for serial scoring
```

Scores are teacher-forced initial-model cross entropy and perplexity, not task
correctness or reward labels. `pretrain_scores.jsonl` contains each source row's
identity, retained/full length, per-turn token spans, truncation flags, valid-token
counts, mean CE and PPL. The companion `.meta.json` describes the scoring scheme.
Source indices are attached before sampling/merging/splitting. A `.incomplete`
file is promoted only after successful completion; existing score files are not
overwritten. Writer errors propagate to all ranks, and evaluation mode/RNG state
are restored before training.

Supported: text-only MCA SFT, TP/PP with CP1/DP1, no virtual PP or MTP, no packing,
streaming, resume, cached tokenization, masking, prompt supervision, row/turn
weighting, long-sample dropping or efficient-EOS templates. For PP with multiple
microbatches enable `variable_seq_lengths`. Score-only runs do not save/export
model weights. Use a fresh output directory.

## Correctness and diagnostics

Text containing literal image/video tokens remains text when no corresponding
media is present in the entire batch. A text-only row containing reserved
image/video token IDs cannot share a batch with corresponding real or dummy
media; the collator rejects this combination before the model's embedding scatter.
Use separate text-only batches with dummy media disabled for frozen vision modules.
Literal reserved IDs inside a row with real media are not supported either.
Frozen MCA vision/projector paths do not insert dummy images.
A collator `StopIteration` becomes an error, so Accelerate cannot silently treat
malformed media as the end of an epoch. Variable-length EP avoids cutoff-length
padding; fixed padding aligns labels, weights and 3D positions on either side.

`LF_MCA_PROFILE=1` adds microbatch progress, forward/backward and optimizer time,
and the maximum allocated/reserved GPU memory across ranks. Synchronization adds
measurement overhead. For each logged step, `input_tokens_per_second` sums input
positions across all microbatches and DP replicas, without double-counting TP/PP/CP
copies, and divides by the longest step interval across ranks. The interval includes
data preparation and any logging, saving or evaluation since the previous step measurement.
Positions are counted after batch padding, before packing and CP partitioning;
this includes padding that packing may later remove. This is an input volume
metric, not a count of supervised tokens or model FLOPs.
`input_tokens_per_second_per_gpu` divides that cluster rate by the total GPU count.
The old adapter throughput key is retained for compatibility.

`output_dir/trainer_log.jsonl` preserves the trainer's logged metrics, including
`grad_norm`, step throughput and, with `LF_MCA_PROFILE=1`, timing and memory metrics.
The existing progress fields and `lr` alias remain available. Metrics are written
at each logging event; they do not require a checkpoint or an external tracker.
This file is a metric history, not a capture of stdout/stderr.

`MEMORY_MONITOR=1` enables host/cgroup/process/GPU memory samples on stdout; it is
optional and exits when the launched process exits. For a reproducible benchmark,
use the same data, length cap and global batch; override `max_steps`, use a fresh
`output_dir`, and set `save_strategy='"no"' benchmark_skip_final_save=true` (quote
`"no"` in YAML). Compare optimizer offload, recomputation and CP choices with the
same common launcher; no parallel duplicate job or run-specific launcher is needed.
Short benchmark runs do not establish longest-sample memory safety.

## HF export

After successful MCA training, node 0 converts the final native checkpoint once,
after workers release their GPUs, into `export_dir` or `output_dir + "-hf"`.
Direct HF saving, benchmark final-save suppression and score-only runs skip this
conversion. `save_hf_model: true` is rejected when the resolved model config has
`mrope_yarn_enabled: true`: direct saving does not finalize YaRN metadata or
compensate legacy weights. This is checked before model allocation and again at
the direct-save entry point. Use `save_hf_model: false` for these checkpoints;
the native-checkpoint conversion below performs both steps. Direct HF saving
remains available without the custom MRoPE/YaRN extension.

Conversion clears inherited distributed ranks. The packaged converter
works from a wheel as well as from a source checkout:

```bash
python -m llamafactory.model.model_utils.megatron_export \
  --checkpoint_path /path/to/native --output_path /path/to/hf --bf16 true
```

`scripts/megatron_merge.py` is a compatibility shim to the same implementation.
Both paths use one metadata finalizer for YaRN factor, original/maximum length,
rotary amplitude, generation settings and tokenizer EOS/PAD/BOS. Automatic export
also validates the tokenizer, checkpoint's chat-template file and safetensors index/shards.
Nonempty or in-place output paths are rejected before conversion.
Streaming export casts floating-point weights to the requested precision (or
the checkpoint's parameter dtype by default), preserves integer tensors, and
records that precision in the root and nested HF model configs.

Export preserves the chat-template file, but does not save `reasoning_effort`,
`enable_thinking` or `preserve_thinking` as tokenizer defaults. Pass the intended
values explicitly when serving Qwen3.8. For example, with a reloaded HF tokenizer:

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Compute 1 + 1."}],
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="low",
    enable_thinking=True,
    preserve_thinking=False,
)
```

Use the same explicit parameters in the LlamaFactory inference config. Omitting
them uses the template defaults, which may differ from the training configuration.

Metadata repair of an existing HF export is available without the CUDA converter:

```bash
python -m llamafactory.model.model_utils.hf_export \
  --checkpoint /path/to/native --output /path/to/hf
```

New training scales only rotary dimensions. Only checkpoints explicitly tagged
`mrope_yarn_mscale_mode: all_dimensions` use legacy behavior. Export compensates
nonrotary Q/K norm weights for those checkpoints (currently verified for dense
Qwen3.5/3.8), supporting both single-file and indexed sharded safetensors, with a
byte journal, interruption recovery and idempotency. Metadata and compensation
inputs are validated before rewriting metadata. Untagged
checkpoints are not guessed to be legacy. Metadata backups and
`hf_export_metadata_report.json` remain local artifacts, not model inputs.

## Validation

The integration check on 2026-11-19 passed **359 selected CPU tests**, with **1
existing expected failure** for the Transformers attention refactor. Ruff found
no new diagnostics; the vendored converter retains its pre-existing lint issues.
Shell syntax checks, YAML topology dry runs and `git diff --check` also passed.
The earlier isolated wheel build verified inclusion
of the in-tree adapter, scoring and export modules.

The CPU suite compares gradients against unsplit references with simulated CP1/2/4,
including accumulated batches with fully masked microbatches. It executes the
installed Megatron loss-dispatch function with mocked collectives and scheduling;
it does not run a distributed PP schedule. Other checks cover template parity against local Qwen
tokenizers, source/turn scoring, masks and EOS, padding and dataset exhaustion,
combined weights, export metadata/legacy norms, direct-save guards, and
launch/export control flow (including argument values and paths with spaces).
Regressions also cover HF gradient accumulation with unequal valid lengths and
partial windows, masked packed segments through the collator and adapter,
per-token metric aggregation, tiny streaming export artifacts and dtype metadata,
config interpolation, and official-template parity for message/tool whitespace
and history cleanup across tool-call chains. Generation tests cover open thinking
prefixes, response/history round trips and prompt construction in all three chat
engines; vLLM and SGLang backend calls are mocked.
The 12 HF gradient-accumulation cases also pass with Transformers 4.55.0.
MCA checks cover MTP validation against the resolved config, padded input counts,
and throughput with unequal DP input lengths and rank timings using mocked collectives.
MoE tests execute the installed auxiliary-loss scaler to verify that trajectory
weights affect only CE gradients.
Set `WANDB_DISABLED=true`; set `QWEN38_TOKENIZER` and `QWEN36_TOKENIZER` to local
checkpoints to include tokenizer parity tests. Otherwise those optional tests skip.
Tests that inspect Megatron source require `megatron-core`; Qwen3.5-specific tests
require a Transformers version containing those model classes. These checks skip
when the optional dependency is absent. POSIX file-lock tests and the Linux launcher
test also skip on unsupported platforms. Use the pinned Linux environment above
to include all of these checks; the ordinary CPU CI matrix covers a subset.

From the repository root, with dependencies installed in `/path/to/env`:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_DISABLED=true HF_HUB_OFFLINE=1 PYTHONPATH=src \
QWEN38_TOKENIZER=/path/to/Qwen3.8-27B QWEN36_TOKENIZER=/path/to/Qwen3.6-27B \
/path/to/env/bin/python -m pytest -q \
  tests/train/test_mca_*.py \
  tests/train/test_log_callback.py \
  tests/train/test_qwen35_fsdp.py \
  tests/train/test_sft_trainer.py::test_compute_weighted_loss \
  tests/train/test_sft_trainer.py::test_uniform_weight_scales_hf_loss_gradient \
  tests/train/test_sft_trainer.py::test_weighted_trainer_gradient_accumulation \
  tests/model/model_utils/test_hf_export.py \
  tests/model/model_utils/test_mscale_compatibility.py \
  tests/model/model_utils/test_rope.py \
  tests/model/model_utils/test_attention.py \
  tests/data/test_converter.py \
  tests/data/test_qwen38_template.py \
  tests/chat/test_generation.py \
  tests/data/processor/test_supervised.py::test_supervised_turn_mask \
  tests/data/processor/test_supervision_controls.py
```

A separate Qwen3.6-27B 256K SFT comparison on two nodes and 16 GPUs reached
39 of 48 steps at the 2026-11-19 inspection (TP4/PP4/CP1, global batch 64).
After removing stale workers from an earlier job, steps 8–39 averaged 0.52%
longer than the baseline. The maximum loss difference was 0.0015; gradient
norms through step 32 differed by up to 15.2%, so this is not a claim of
stepwise numerical equivalence. Checkpoints at steps 16 and 32 passed shard
header, index and file-length checks, without a full model reload.
This run used cuDNN 9.10.2.21 with FlashAttention enabled and cuDNN attention
disabled; it does not validate cuDNN fused attention or CP packing.

Completed full-length training, 512K execution, complete native-checkpoint
conversion and serving evaluation still require separate acceptance checks.
A fresh CUDA environment installation was not tested. CPU and packaging
results alone do not certify those paths.
