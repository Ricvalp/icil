#!/usr/bin/env bash
#SBATCH --job-name=quickdraw_bc5_heldout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Prefer submit_quickdraw_h200.sh bc5-heldout for H200 resource detection.
set -euo pipefail
case "${1:-bc5-heldout}" in
    bc5-heldout) if (( $# )); then shift; fi ;;
    --*) ;;
    *) printf 'Pass optional trainer arguments after bc5-heldout.\n' >&2; exit 2 ;;
esac
cd -- "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
exec bash hpc/quickdraw_h200.sbatch bc5-heldout "$@"
