#!/usr/bin/env bash
# Simulate a DaPPA pipeline in uPIMulator from a recipe emitted with
# DAPPA_EMIT_RECIPE, then print the end-to-end cost model estimate.
#
# Usage: ./run.sh RECIPE_DIR [UPIMULATOR_DIR]
#
#   RECIPE_DIR       directory written by DAPPA_EMIT_RECIPE=<dir>
#   UPIMULATOR_DIR   uPIMulator checkout prepared by setup.sh
#                    (default: integrations/upimulator/uPIMulator)
#
# The simulator log is written to RECIPE_DIR/bin/log.txt.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$(cd "$1" && pwd)"
SIM="$(cd "${2:-$HERE/uPIMulator}" && pwd)/golang/uPIMulator"

if [[ ! -f "$RECIPE/recipe.json" || ! -f "$RECIPE/device.c" ]]; then
    echo "error: $RECIPE does not contain recipe.json and device.c" >&2
    exit 1
fi
if [[ ! -x "$SIM/build/uPIMulator" ]]; then
    echo "error: $SIM/build/uPIMulator not found; run setup.sh first" >&2
    exit 1
fi

read -r DPUS TASKLETS < <(python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))['pipeline']
print(p['dpu_count'], p['num_tasklets'])" "$RECIPE/recipe.json")

# The DAPPA benchmark compiles whatever device code sits in its task.c.
cp "$RECIPE/device.c" "$SIM/benchmark/DAPPA/dpu/task.c"

BIN="$RECIPE/bin"
rm -rf "$BIN" && mkdir -p "$BIN"

cd "$SIM"
DAPPA_RECIPE_DIR="$RECIPE" DAPPA_SKIP_VERIFY=1 ./build/uPIMulator \
    --root_dirpath "$SIM" \
    --bin_dirpath "$BIN" \
    --benchmark DAPPA \
    --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank "$DPUS" \
    --num_tasklets "$TASKLETS" \
    --data_prep_params 1

python3 "$HERE/cost_model.py" "$RECIPE" "$BIN"
