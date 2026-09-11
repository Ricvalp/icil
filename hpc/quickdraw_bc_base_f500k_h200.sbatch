#!/usr/bin/env bash
#SBATCH --job-name=quickdraw_bc_base_f500k
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Prefer submit_quickdraw_h200.sh bc-base-f500k for H200 resource detection.
set -euo pipefail
case "${1:-bc-base-f500k}" in
    bc-base-f500k) if (( $# )); then shift; fi ;;
    --*) ;;
    *) printf 'Pass optional trainer arguments after bc-base-f500k.\n' >&2; exit 2 ;;
esac
# These larger fast states default to the CUDA-graph workaround used on peano.
# Set QUICKDRAW_CUDA_GRAPHS=1 to preserve inherited XLA flags without this override.
case "${QUICKDRAW_CUDA_GRAPHS:-0}" in
    0) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=" ;;
    1) ;;
    *) printf 'QUICKDRAW_CUDA_GRAPHS must be 0 or 1.\n' >&2; exit 2 ;;
esac
printf 'QUICKDRAW_CUDA_GRAPHS=%s XLA_FLAGS=%s\n' "${QUICKDRAW_CUDA_GRAPHS:-0}" "${XLA_FLAGS:-}"
cd -- "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
exec bash hpc/quickdraw_h200.sbatch bc-base-f500k "$@"
