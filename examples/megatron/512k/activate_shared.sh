# Source with an environment prefix; works with conda environments and virtualenvs.
_lf_activate_mcore() {
    local env_path=${1:-${VENV_PATH:-}}
    if [[ -z $env_path || ! -x $env_path/bin/python ]]; then
        printf 'Usage: source examples/megatron/512k/activate_shared.sh /env-prefix\n' >&2
        return 2
    fi
    local repo_dir library_paths
    env_path=$(realpath -e -- "$env_path") || return
    repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd) || return
    export PATH="$env_path/bin:$PATH"
    export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
    library_paths=$("$env_path/bin/python" -c 'import pathlib,site,sys; paths=[pathlib.Path(sys.prefix)/"lib"]; paths.extend(p for b in site.getsitepackages() for p in pathlib.Path(b).glob("nvidia/*/lib")); print(":".join(str(p) for p in paths if p.is_dir()))') || return
    export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export USE_MCA=1
    export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}
_lf_activate_mcore "$@"
