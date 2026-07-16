"""Upload final SFT model (weights + tokenizer/config) to Hugging Face.

Uses upload_large_folder for concurrent multi-worker upload with built-in
resume, retry, and per-file commit. Combined with hf_transfer for multi-part
parallel PUT on the LFS path.

Notes:
- HuggingFace's Xet backend (cas-server.xethub.hf.co) can have high latency in
  some regions, causing xorb uploads to timeout.
  HF_HUB_DISABLE_XET=1 forces the legacy LFS path which goes through regional PoPs.
- hf_transfer accelerates LFS uploads via Rust multi-part parallel PUT. It is only
  deprecated for the Xet path; for LFS it still works and helps significantly.
- DO NOT use AutoModelForCausalLM.from_pretrained + save_pretrained to re-shard
  multimodal models — it drops the vision encoder and rewrites config.json.
  Re-shard directly via the safetensors library to preserve weights bit-for-bit.
"""

import argparse
import os
import sys
from pathlib import Path

# Force the legacy LFS path when Xet's CAS server has high regional latency.
os.environ["HF_HUB_DISABLE_XET"] = "1"
# hf_transfer accelerates LFS uploads via multi-part parallel PUT.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Reuse the standalone resharder in tools/.
sys.path.insert(0, str(Path(__file__).parent / "tools"))
import reshard_safetensors as _reshard_tool  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Defaults — overridable via CLI args
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CKPT_DIR = os.environ.get("CKPT_DIR")
DEFAULT_REPO_ID = os.environ.get("HF_REPO_ID")
DEFAULT_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_MAX_SHARD_SIZE = "5GB"
DEFAULT_NUM_WORKERS = 16
# ──────────────────────────────────────────────────────────────────────────────

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt"]

IGNORE_PATTERNS = [
    "training_args.bin",
    "trainer_state.json",
    "trainer_log.jsonl",
    "all_results.json",
    "train_results.json",
    "training_loss.png",
    "README.md",
    "checkpoint-*",
    "global_step*",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload model checkpoint to Hugging Face")
    p.add_argument("ckpt_dir", nargs="?", default=DEFAULT_CKPT_DIR, help="Local checkpoint directory (or CKPT_DIR)")
    p.add_argument("repo_id", nargs="?", default=DEFAULT_REPO_ID, help="HF repo id (org/name, or HF_REPO_ID)")
    p.add_argument("--token", default=DEFAULT_TOKEN, help="HF API token (defaults to HF_TOKEN)")
    p.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE, help="Max shard size (e.g. 5GB)")
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of upload workers")
    p.add_argument("--private", action="store_true", default=True, help="Create private repo (default)")
    p.add_argument("--public", action="store_true", help="Create public repo")
    args = p.parse_args()
    if not args.ckpt_dir:
        p.error("ckpt_dir is required (pass it as an argument or set CKPT_DIR)")
    if not args.repo_id:
        p.error("repo_id is required (pass it as an argument or set HF_REPO_ID)")
    if not args.token:
        p.error("a Hugging Face token is required (pass --token or set HF_TOKEN)")
    return args


def reshard_if_needed(src_dir: str, max_shard_size: str) -> str:
    """If any safetensors file > max_shard_size, re-shard into a tagged output dir."""
    import glob

    threshold_bytes = _reshard_tool.parse_size(max_shard_size)
    safetensors_files = sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors")))

    needs_reshard = any(os.path.getsize(f) > threshold_bytes for f in safetensors_files)
    if not needs_reshard:
        print(f"All shards already <= {max_shard_size}, no re-sharding needed.")
        return src_dir

    # Tag directory with shard size to avoid reusing stale shards from a different config.
    size_tag = max_shard_size.replace(" ", "").upper()
    dst_dir = src_dir.rstrip("/") + f"_sharded_{size_tag}"
    if os.path.exists(dst_dir) and glob.glob(os.path.join(dst_dir, "model-*.safetensors")):
        print(f"Sharded directory already exists: {dst_dir}")
        return dst_dir

    print(f"Re-sharding via tools/reshard_safetensors.py (max_shard_size={max_shard_size})...")
    _reshard_tool.reshard(src_dir, dst_dir, threshold_bytes)
    return dst_dir


def verify_upload(api: HfApi, repo_id: str, local_dir: str) -> bool:
    """Compare remote file list against local to confirm upload completeness."""
    remote_files = {}
    for info in api.list_repo_tree(repo_id=repo_id, repo_type="model", recursive=False):
        if hasattr(info, "size") and info.size is not None:
            remote_files[info.path] = info.size

    local_files = {}
    for f in os.listdir(local_dir):
        filepath = os.path.join(local_dir, f)
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(f)[1]
        if ext not in {".safetensors", ".json", ".jinja", ".txt"}:
            continue
        if f in {"training_args.bin", "trainer_state.json", "trainer_log.jsonl",
                 "all_results.json", "train_results.json", "training_loss.png", "README.md"}:
            continue
        local_files[f] = os.path.getsize(filepath)

    ok = True
    for name, size in local_files.items():
        remote_size = remote_files.get(name)
        if remote_size is None:
            print(f"  MISSING on remote: {name} ({size / 1024**3:.2f} GB)")
            ok = False
        elif remote_size != size:
            print(f"  SIZE MISMATCH: {name} (local={size}, remote={remote_size})")
            ok = False

    if ok:
        total_gb = sum(local_files.values()) / 1024**3
        print(f"  Verified: {len(local_files)} files, {total_gb:.1f} GB total — all match.")
    return ok


def main() -> int:
    args = parse_args()
    private = not args.public

    print(f"CKPT_DIR = {args.ckpt_dir}")
    print(f"REPO_ID  = {args.repo_id}")
    print(f"Workers  = {args.num_workers}, MaxShard = {args.max_shard_size}")

    api = HfApi(token=args.token)

    print(f"Creating repo {args.repo_id} (private={private})...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=private)

    upload_dir = reshard_if_needed(args.ckpt_dir, args.max_shard_size)

    print(f"Uploading with {args.num_workers} workers (hf_transfer + LFS)...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=upload_dir,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        num_workers=args.num_workers,
    )

    print("\nVerifying upload...")
    if not verify_upload(api, args.repo_id, upload_dir):
        print("WARNING: verification failed — some files may need re-upload.")
        return 1

    # Clean up sharded directory if it was created (different from source).
    if upload_dir != args.ckpt_dir:
        import shutil
        print(f"Cleaning up sharded directory: {upload_dir}")
        shutil.rmtree(upload_dir)

    print(f"\nDone! https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
