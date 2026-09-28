# Source this file to use the shared Megatron 512K environment (conda env `lf`).
# It does not launch training or change any dataset/training configuration.
_programgen_activate_shared_mcore() {
    local mcore_conda_root=/public/storage/conghao/miniconda
    local mcore_repo_root=/public/storage/conghao/LLaMA-Factory
    local mcore_library_paths
    source "$mcore_conda_root/etc/profile.d/conda.sh" || return
    conda activate "$mcore_conda_root/envs/lf" || return
    export CUDA_HOME="$CONDA_PREFIX"
    export PYTHONPATH="$mcore_repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
    mcore_library_paths=$(python -B -c 'import pathlib,site,sys; paths=[pathlib.Path(sys.prefix)/"lib"]; paths.extend(p for b in site.getsitepackages() for p in pathlib.Path(b).glob("nvidia/*/lib")); print(":".join(str(p) for p in paths if p.is_dir()))')
    export LD_LIBRARY_PATH="$mcore_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export USE_MCA=1
    export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/lf/torch_extensions}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lf/triton}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/lf/inductor}"
}
_programgen_activate_shared_mcore
unset -f _programgen_activate_shared_mcore
