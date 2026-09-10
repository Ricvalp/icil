#!/usr/bin/env bash
# Detect the site's registered H200 GRES or feature before submitting one job.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
mode="${1:-ar}"
case "$mode" in
    ar|autoregressive|diffusion|kvb) batch_script=hpc/quickdraw_h200.sbatch ;;
    ar-heldout|diffusion-heldout) batch_script=hpc/quickdraw_icil_heldout_h200.sbatch ;;
    kvb-heldout) batch_script=hpc/quickdraw_kvb_heldout_h200.sbatch ;;
    bc-heldout) batch_script=hpc/quickdraw_bc_heldout_h200.sbatch ;;
    bc1-heldout) batch_script=hpc/quickdraw_bc1_heldout_h200.sbatch ;;
    bc5-heldout) batch_script=hpc/quickdraw_bc5_heldout_h200.sbatch ;;
    bc-fast128-heldout) batch_script=hpc/quickdraw_bc_fast128_heldout_h200.sbatch ;;
    bc-large-heldout) batch_script=hpc/quickdraw_bc_large_heldout_h200.sbatch ;;
    kvb-large-heldout) batch_script=hpc/quickdraw_kvb_large_heldout_h200.sbatch ;;
    visualize|fid) batch_script=hpc/quickdraw_evaluate_h200.sbatch ;;
    *) printf 'Unknown experiment: %s. See hpc/README.md for submission modes.\n' "$mode" >&2; exit 2 ;;
esac

gpu_arguments_text="$(sinfo --noheader --partition=gpuq --format='%G|%f' | python3 -c '
import re
import sys
rows = [line.strip().split("|", 1) for line in sys.stdin if line.strip()]
types = {kind for row in rows for kind in re.findall(r"gpu:([^:(),\s]+):\d+", row[0])
         if "h200" in kind.lower()}
features = {feature.strip() for row in rows if len(row) == 2
            for feature in row[1].split(",") if "h200" in feature.lower()}
if len(types) == 1:
    print("--gres=gpu:" + next(iter(types)) + ":1")
elif not types and len(features) == 1:
    print("--gres=gpu:1")
    print("--constraint=" + next(iter(features)))
else:
    raise SystemExit("Cannot identify one unambiguous H200 resource in gpuq. "
                     "Inspect sinfo -p gpuq -o \"%G %f\" and supply the site GRES/constraint "
                     "explicitly when submitting hpc/quickdraw_h200.sbatch.")
')"
mapfile -t gpu_arguments <<< "$gpu_arguments_text"
mkdir -p hpc/logs
printf 'Submitting one 12-hour H200 job: %s\n' "${gpu_arguments[*]}"
exec sbatch "${gpu_arguments[@]}" --job-name="quickdraw_${mode}" \
    "$batch_script" "$@"
