#!/usr/bin/env bash
#SBATCH --job-name=quickdraw_icil_heldout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Prefer submit_quickdraw_h200.sh ar-heldout or diffusion-heldout so the
# cluster's H200 GRES spelling is detected. Direct sbatch defaults to AR.
set -euo pipefail
mode=ar-heldout
case "${1:-ar}" in
    ar|autoregressive|ar-heldout) if (( $# )); then shift; fi ;;
    diffusion|diffusion-heldout) mode=diffusion-heldout; shift ;;
    --*) ;;
    *) printf 'Use ar or diffusion, followed by optional trainer arguments.\n' >&2; exit 2 ;;
esac
cd -- "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
exec bash hpc/quickdraw_h200.sbatch "$mode" "$@"
