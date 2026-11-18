import hashlib
import json
import os
import pathlib
import sys
import time


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aligned_train import GPUAlignedTrainer
from loading import install_loaded_leaf_guard
from transformers import TrainerCallback


install_loaded_leaf_guard()
if os.getenv("PROBE_CHECKPOINT_CPU_CACHE") == "1":
    from checkpoint_storage import install_checkpoint_cpu_cache

    install_checkpoint_cpu_cache()
import faulthandler  # noqa: E402 - preserve runtime patch initialization order
import signal  # noqa: E402 - preserve runtime patch initialization order


faulthandler.register(signal.SIGUSR1, all_threads=True)


class Probe(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if os.getenv("PROBE_CPU_OFFLOAD_LOAD") == "1":
            kwargs["model"].cpu()
            for module in kwargs["model"].modules():
                for name, buffer in module._buffers.items():
                    if buffer is not None:
                        module._buffers[name] = buffer.to(args.device)
            assert all(p.device.type == "cpu" for p in kwargs["model"].parameters())
            if state.is_world_process_zero:
                print("CPU_OFFLOAD_PARAMETERS_MATERIALIZED", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            logs = dict(logs or {})
            if "learning_rate" in logs:
                logs["learning_rate_used"] = logs["learning_rate"]
                logs["learning_rate_gpu_convention"] = kwargs["optimizer"].param_groups[0]["lr"]
            with open(pathlib.Path(args.output_dir) / "probe-metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")


class RealTrainer(GPUAlignedTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.add_callback(Probe())
        self.micro = 0
        import torch

        if os.getenv("PROBE_VERIFY_WEIGHTS") == "1" and self.args.process_index == 0:
            from loading import verify_loaded_weights

            verify_loaded_weights(self.model, pathlib.Path(self.args.output_dir) / "initial-weights-verification.json")
        if os.getenv("PROBE_CHUNK_CE") == "1":
            from memory import patch_chunked_ce

            patch_chunked_ce(self.model)
        if int(os.getenv("PROBE_GDN_SEGMENT", "0")):
            from segmented_gdn import install_segmented_gdn

            count = install_segmented_gdn(self.model, int(os.environ["PROBE_GDN_SEGMENT"]))
            if self.args.process_index == 0:
                print(f"SEGMENTED_GDN layers={count} segment_tokens={os.environ['PROBE_GDN_SEGMENT']}", flush=True)
        if int(os.getenv("PROBE_MLP_CHUNK", "0")):
            from chunked_mlp import install_chunked_mlp

            install_chunked_mlp(self.model, int(os.environ["PROBE_MLP_CHUNK"]))
        if int(os.getenv("PROBE_NORM_CHUNK", "0")):
            from chunked_norm import install_chunked_norm

            install_chunked_norm(self.model, int(os.environ["PROBE_NORM_CHUNK"]))
        if int(os.getenv("PROBE_ATTN_CHUNK", "0")):
            from chunked_attention_ops import install_chunked_attention_ops

            install_chunked_attention_ops(self.model, int(os.environ["PROBE_ATTN_CHUNK"]))
        if os.getenv("PROBE_NESTED_CHECKPOINT") == "1":
            from nested_checkpoint import install_nested_checkpoint

            count = install_nested_checkpoint(self.model)
            if self.args.process_index == 0:
                print(f"NESTED_CHECKPOINT modules={count}", flush=True)
        if os.getenv("PROBE_RECOMPUTE_OFFLOAD") == "1":
            from recompute_offload import install_recompute_offload

            install_recompute_offload(self.model)
        if os.getenv("PROBE_LAYER_TRACE", "1") == "1":
            path = pathlib.Path(self.args.output_dir) / f"layers-rank{self.args.process_index}.jsonl"

            def hook(index, phase):
                def record(module, args, output=None):
                    if self.state.global_step < int(os.getenv("PROBE_LAYER_TRACE_STEPS", "2")):
                        with path.open("a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "time": time.time(),
                                        "layer": index,
                                        "phase": phase,
                                        "step": self.state.global_step,
                                        "allocated_GiB": torch.npu.memory_allocated() / 2**30,
                                    }
                                )
                                + "\n"
                            )

                return record

            for index, layer in enumerate(self.model.model.language_model.layers):
                layer.register_forward_pre_hook(hook(index, "enter"))
                layer.register_forward_hook(hook(index, "exit"))
        from torch.distributed.fsdp import MixedPrecisionPolicy

        self.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None
        )
        from cpu_offload_clip import clip_sharded_grad_norm

        self.accelerator.clip_grad_norm_ = lambda parameters, max_norm, norm_type=2: clip_sharded_grad_norm(
            parameters, max_norm, self.accelerator.device, norm_type
        )
        if os.getenv("PROBE_GRAD_AUDIT") == "1":
            assert sum(p.numel() for p in self.model.parameters()) < 10_000_000, "Gradient dump is tiny-model only"
            original_clip = self.accelerator.clip_grad_norm_

            def audited_clip(parameters, max_norm, norm_type=2):
                torch.npu.synchronize()
                if self.state.global_step == 0:
                    grads = {
                        name: (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).detach().cpu().clone()
                        for name, p in self.model.named_parameters()
                        if p.grad is not None
                    }
                    torch.save(
                        grads, pathlib.Path(self.args.output_dir) / f"gradients-rank{self.args.process_index}.pt"
                    )
                return original_clip(parameters, max_norm, norm_type)

            self.accelerator.clip_grad_norm_ = audited_clip
        # Reduce every microbatch so full unsharded 27B gradients do not accumulate in HBM.
        self.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

    def compute_loss(self, model, inputs, **kw):
        import torch

        valid_length = int(inputs["attention_mask"][0].sum())
        input_sha = hashlib.sha256(
            inputs["input_ids"][0, :valid_length].detach().cpu().numpy().astype("<i4").tobytes()
        ).hexdigest()
        labels_sha = hashlib.sha256(
            inputs["labels"][0, :valid_length].detach().cpu().numpy().astype("<i4").tobytes()
        ).hexdigest()
        start = time.monotonic()
        loss = super().compute_loss(model, inputs, **kw)
        record = {
            "step": self.state.global_step,
            "micro": self.micro,
            "rank": self.args.process_index,
            "length": inputs["input_ids"].shape[-1],
            "loss": float(loss.detach()),
            "forward_seconds": time.monotonic() - start,
            "peak_GiB": torch.npu.max_memory_allocated() / 2**30,
        }
        record["valid_length"] = valid_length
        record["supervised_tokens"] = int((inputs["labels"] != -100).sum())
        record["input_sha256"] = input_sha
        record["labels_sha256"] = labels_sha
        with open(pathlib.Path(self.args.output_dir) / f"micro-rank{self.args.process_index}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self.micro += 1
        return loss

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint and os.getenv("PROBE_CPU_OFFLOAD_LOAD") == "1":
            self.model.cpu()
            for module in self.model.modules():
                for name, buffer in module._buffers.items():
                    if buffer is not None:
                        module._buffers[name] = buffer.to(self.args.device)
        return super()._load_optimizer_and_scheduler(checkpoint)

    def save_model(self, *a, **kw):
        if os.getenv("PROBE_SAVE_MODEL") == "1":
            return super().save_model(*a, **kw)
        # Short diagnostic runs only write metrics.


if os.getenv("PROBE_ACTIVATION_OFFLOAD") == "1":
    import torch.utils.checkpoint
    from memory import offload_checkpoint

    torch.utils.checkpoint.checkpoint = offload_checkpoint
import llamafactory.train.sft.workflow as workflow  # noqa: E402 - preserve runtime patch initialization order
from llamafactory.train.tuner import run_exp  # noqa: E402 - preserve runtime patch initialization order


if __name__ == "__main__":
    workflow.CustomSeq2SeqTrainer = RealTrainer
    run_exp()
