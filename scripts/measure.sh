#!/usr/bin/env bash
# Run a group of tests repeatedly and collect the per-stage timings they print.
#
# Usage: scripts/measure.sh WARMUP_ROUNDS MEASUREMENT_ROUNDS BENCHMARK
#   BENCHMARK is the test name prefix before the colon, e.g. "VA" or "HST-S".
#   Timings are only printed when MEASURE_PERFORMANCE_HOST and/or
#   MEASURE_PERFORMANCE_DPU are enabled in include/logging.h.
#   The test binary is taken from $BUILD_DIR (default: build/release).
#   Results are written to ./measurement_fw_<benchmark>.txt, one row per round.

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 WARMUP_ROUNDS MEASUREMENT_ROUNDS BENCHMARK"
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-build/release}"
[[ "$build_dir" = /* ]] || build_dir="$repo_root/$build_dir"
tests="$build_dir/tests/tests"
if [[ ! -x "$tests" ]]; then
  echo "Test binary not found: $tests"
  exit 1
fi

# Warmup
for i in $(seq "$1"); do
    echo -en "\rRunning warmup round $i..."
    "$tests" -r compact "$3: *" > /dev/null
done
if [[ "$1" -ne 0 ]]; then
    echo " Done"
fi

# Measurements
name="measurement_fw_$(echo "$3" | cut -f1 -d':' | cut -f1 -d'*' | tr '[:upper:]' '[:lower:]').txt"
key=""
echo -en "" > "$name"
for i in $(seq "$2"); do
    echo -en "\rRunning measurement round $i..."
    value=""
    while read -r line; do
        if echo "$line" | grep "passed: " > /dev/null; then
            break
        fi
        value="$value\t$(echo $line | cut -f2 -d':' | cut -f2 -d ' ' | sed 's/ms//g')"
        if [[ $i -eq 1 ]]; then
            key="$key\t$(echo $line | cut -f1 -d':')"
        fi
    done < <("$tests" -r compact "$3: *")
    if [[ $i -eq 1 ]]; then
        echo -e "${key:2}" >> "$name"
    fi
    echo -e "${value:2}" >> "$name"
done
if [[ "$2" -ne 0 ]]; then
    echo " Done"
fi
