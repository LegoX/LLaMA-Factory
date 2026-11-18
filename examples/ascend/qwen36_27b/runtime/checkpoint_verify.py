"""Structural DCP checkpoint checks without loading full model/optimizer tensors."""

import json
from pathlib import Path


def verify_dcp_dir(path):
    from torch.distributed.checkpoint import FileSystemReader

    path = Path(path)
    if not (path / ".metadata").is_file():
        raise ValueError(f"Missing DCP metadata: {path}")
    metadata = FileSystemReader(path).read_metadata()
    if not metadata.state_dict_metadata or not metadata.storage_data:
        raise ValueError(f"Empty DCP checkpoint: {path}")
    files = {}
    for location in metadata.storage_data.values():
        shard = (path / location.relative_path).resolve()
        if not shard.is_relative_to(path.resolve()):
            raise ValueError("Checkpoint references an external file")
        size = shard.stat().st_size
        if location.offset < 0 or location.length < 0 or location.offset + location.length > size:
            raise ValueError(f"Truncated DCP shard: {shard}")
        files[shard.name] = size
    return {"path": str(path), "state_entries": len(metadata.state_dict_metadata), "shard_files": files}


def verify_checkpoint(path, world_size=16):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    if not isinstance(state.get("global_step"), int) or state["global_step"] < 1:
        raise ValueError("Invalid trainer step")
    for name in ["scheduler.pt"] + [f"rng_state_{rank}.pth" for rank in range(world_size)]:
        if not (path / name).is_file() or not (path / name).stat().st_size:
            raise ValueError(f"Missing or empty checkpoint file: {name}")
    return {
        "step": state["global_step"],
        "world_size": world_size,
        "model": verify_dcp_dir(path / "pytorch_model_fsdp_0"),
        "optimizer": verify_dcp_dir(path / "optimizer_0"),
        "scope": "Checks metadata and referenced byte ranges; does not replace a full restore test.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--world-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint, args.world_size), indent=2))
