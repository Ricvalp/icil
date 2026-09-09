#!/usr/bin/env bash
#SBATCH --job-name=quickdraw_evaluate
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# The submission helper resolves the cluster's H200 GRES spelling.
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"

mode="${1:-}"
case "$mode" in
    visualize) module=supervised_visualize; command=() ;;
    fid) module=supervised_fid; command=(evaluate) ;;
    *) printf 'Use visualize or fid, followed by checkpoint evaluation arguments.\n' >&2; exit 2 ;;
esac
shift

dataset_root="${QUICKDRAW_DATASET_ROOT:-/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1}"
if [[ ! -x .venv/bin/python || ! -f "icil_jax_rlbench/quickdraw/$module.py" ]]; then
    printf 'Missing environment or code. Follow hpc/README.md before submitting.\n' >&2
    exit 1
fi
if [[ ! -f "$dataset_root/manifest.json" ]]; then
    printf 'Dataset manifest not found: %s/manifest.json\n' "$dataset_root" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

# Refuse a CPU fallback or an accidental allocation of another GPU model.
srun --ntasks=1 .venv/bin/python -u - <<'PY'
import jax
devices = jax.local_devices()
if len(devices) != 1 or devices[0].platform != 'gpu' or 'H200' not in devices[0].device_kind.upper():
    raise SystemExit(f'Expected exactly one H200; JAX sees {devices!r}')
print(f'JAX {jax.__version__}: {devices[0].device_kind}', flush=True)
PY

exec srun --ntasks=1 .venv/bin/python -u \
    -m "icil_jax_rlbench.quickdraw.$module" "${command[@]}" \
    --dataset-root "$dataset_root" "$@"
