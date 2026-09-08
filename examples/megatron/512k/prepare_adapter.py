import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_adapter(adapter, apply=False):
    adapter = Path(adapter).resolve()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    patch = BUNDLE / "mcore-adapter.patch"
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("Patch checksum mismatch")
    head = subprocess.check_output(["git", "-C", str(adapter), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["base_commit"]:
        raise ValueError(f"Expected adapter baseline {manifest['base_commit']}, got {head}")
    changed = set(
        subprocess.check_output(["git", "-C", str(adapter), "diff", "HEAD", "--name-only"], text=True).splitlines()
    )
    if changed - manifest["files"].keys():
        raise ValueError("Adapter has unrelated tracked modifications; use a separate clean checkout")
    hashes = {name: digest(adapter / name) for name in manifest["files"]}
    patched = all(hashes[name] == values["patched_sha256"] for name, values in manifest["files"].items())
    if not patched:
        clean = all(hashes[name] == values["base_sha256"] for name, values in manifest["files"].items())
        if not clean:
            raise ValueError("Adapter files are neither the exact baseline nor the exact packaged patch")
        if not apply:
            raise ValueError("Patch is not applied; use --apply on a separate clean adapter checkout")
        subprocess.run(["git", "-C", str(adapter), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(adapter), "apply", str(patch)], check=True)
    for name, values in manifest["files"].items():
        if digest(adapter / name) != values["patched_sha256"]:
            raise ValueError(f"Patched file checksum mismatch: {name}")
    repository = BUNDLE.parents[2]
    for name, expected in manifest["llamafactory_files"].items():
        if digest(repository / name) != expected:
            raise ValueError(f"Missing or different LLaMA-Factory integration: {name}")
    print("Verified adapter baseline, six patched files, and LLaMA-Factory integration")


def main():
    parser = argparse.ArgumentParser(
        description="Apply or verify the exact packaged MCA patch; never installs packages"
    )
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply only to the exact unmodified baseline")
    args = parser.parse_args()
    verify_adapter(args.adapter, args.apply)


if __name__ == "__main__":
    main()
