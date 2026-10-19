#!/usr/bin/env bash
# Cost a DaPPA pipeline on SIMDRAM with the processing-using-DRAM backend.
#
# Usage: BBOP_UTIL_DIR=<dir> ./run.sh OUT_DIR -- PROGRAM [ARGS...]
#
#   BBOP_UTIL_DIR   directory with bbop_manager.c and bbop_manager.h
#   OUT_DIR         where harness.c, the harness binary and bbop_statistics.csv go
#   PROGRAM         any DaPPA program, e.g. build/release/tests/tests "group: PUD parametric"
#
# Prints the SIMDRAM_64 compute latency (ms) reported by the cost model.

set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" || -z "${BBOP_UTIL_DIR:-}" ]]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

OUT="$1"
shift 2
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# 1. Run the DaPPA program; execute() writes OUT/harness.c and returns.
DAPPA_EMIT_BBOP="$OUT" "$@" > /dev/null

# 2. Build and run the harness against the cost model.
gcc "$OUT/harness.c" "$BBOP_UTIL_DIR/bbop_manager.c" -I"$BBOP_UTIL_DIR" \
    -DBBOP_SKIP_COMPUTE -O2 -fopenmp -lm -o "$OUT/harness"
(cd "$OUT" && ./harness > harness.log)

# 3. Report the SIMDRAM_64 summary line (field 6 is the latency in ms).
grep '^Summary, , , , SIMDRAM_64,' "$OUT/bbop_statistics.csv" | cut -d, -f6 | tr -d ' '
