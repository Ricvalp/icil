#!/usr/bin/env bash
#SBATCH --job-name=quickdraw
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Submit from the repository root, preferably through submit_quickdraw_h200.sh,
# which replaces the default GRES spelling with this cluster's H200 spelling.
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"

architecture="${1:-ar}"
if (( $# )); then shift; fi
case "$architecture" in
    ar|autoregressive) config=icil_jax_rlbench/configs/quickdraw_ar_transformer.py ;;
    diffusion) config=icil_jax_rlbench/configs/quickdraw_diffusion_transformer.py ;;
    *) printf 'Use ar or diffusion, followed by optional trainer arguments.\n' >&2; exit 2 ;;
esac

if [[ ! -x .venv/bin/python || ! -f "$config" || ! -f datasets/quickdraw_full_nn_v1/manifest.json ]]; then
    printf 'Missing environment, code, or dataset. Follow hpc/README.md before submitting.\n' >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
wandb_mode="${WANDB_MODE:-online}"
case "$wandb_mode" in
    online|offline|disabled) ;;
    *) printf 'WANDB_MODE must be online, offline, or disabled.\n' >&2; exit 2 ;;
esac

# Refuse a CPU fallback or an accidental allocation of another GPU model.
srun --ntasks=1 .venv/bin/python -u - <<'PY'
import jax
devices = jax.local_devices()
if len(devices) != 1 or devices[0].platform != 'gpu' or 'H200' not in devices[0].device_kind.upper():
    raise SystemExit(f'Expected exactly one H200; JAX sees {devices!r}')
print(f'JAX {jax.__version__}: {devices[0].device_kind}', flush=True)
PY

# No package installation on the compute node. Extra --set overrides and
# --resume pass through to the trainer; existing hyperparameters stay intact.
exec srun --ntasks=1 .venv/bin/python -u \
    -m icil_jax_rlbench.quickdraw.supervised_train \
    --config "$config" --set "wandb_mode=\"$wandb_mode\"" "$@"
