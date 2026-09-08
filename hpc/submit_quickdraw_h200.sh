#!/usr/bin/env bash
# Detect the site's registered H200 GRES or feature before submitting one job.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
architecture="${1:-ar}"
case "$architecture" in
    ar|autoregressive|diffusion) ;;
    *) printf 'Usage: bash hpc/submit_quickdraw_h200.sh [ar|diffusion] [trainer arguments]\n' >&2; exit 2 ;;
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
exec sbatch "${gpu_arguments[@]}" --job-name="quickdraw_${architecture}" \
    hpc/quickdraw_h200.sbatch "$@"
