#!/usr/bin/env python3
"""
End-to-end execution time estimate for a DaPPA pipeline simulated with uPIMulator.

Inputs:
  recipe_dir  Directory written by a DaPPA program run with DAPPA_EMIT_RECIPE=<dir>
              (recipe.json holds the per-round transfer sizes).
  bin_dir     uPIMulator output directory (bin/log.txt holds the DPU cycle counts).

Output: an estimate of the end-to-end time, broken down into DPU allocation,
CPU-DPU transfers, DPU kernel time, DPU-CPU transfers, and host post-processing
(filter compaction, reduction combine, scan stitching). Use --json for
machine-readable output.

Usage:
  python3 cost_model.py <recipe_dir> <bin_dir> [--frequency-mhz 350] [--json]

The transfer coefficients follow the CPU-DPU bandwidth measurements of the PrIM
paper (Gomez-Luna et al., IEEE Access 2022, Figs. 14-16).
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Coefficients (PrIM paper, Section 3.3 and Figs. 14-16)
# ---------------------------------------------------------------------------

COEFFS = {
    # One-time DPU allocation by the UPMEM SDK.
    "dpu_alloc_s": 1.2,

    # CPU-DPU parallel transfers (push and broadcast): fixed per-call latency
    # (alpha) plus bytes / sustained bandwidth (beta). PrIM Fig. 14.
    "cpu_to_dpu_alpha_s": 100e-6,        # 100 us per call
    "cpu_to_dpu_beta_Bps":  6.0e9,       # 6 GB/s per rank

    # DPU-CPU parallel transfers, slightly slower than CPU-DPU. PrIM Fig. 15.
    "dpu_to_cpu_alpha_s": 100e-6,
    "dpu_to_cpu_beta_Bps":  5.0e9,       # 5 GB/s per rank

    # Host post-processing: filter compaction is memmove-bound; reductions cost
    # one combiner call per partial result.
    "host_memmove_Bps":    8.0e9,        # 8 GB/s memmove
    "host_combine_op_s":   2e-9,         # 2 ns per combiner call
}

DEFAULT_DPU_FREQ_MHZ = 350  # uPIMulator default and UPMEM DPU clock


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_recipe(recipe_dir: Path) -> dict:
    path = recipe_dir / "recipe.json"
    if not path.is_file():
        sys.exit(f"recipe.json not found at {path}")
    with path.open() as f:
        return json.load(f)


def load_dpu_cycles(bin_dir: Path) -> tuple[int, int]:
    """Return (max logic cycles over all DPUs, total instructions) from log.txt.

    DPUs run in parallel, so the kernel time is set by the slowest DPU.
    """
    log_path = bin_dir / "log.txt"
    if not log_path.is_file():
        sys.exit(f"log.txt not found at {log_path}")

    cycle_re = re.compile(r"Logic\[(\d+)_(\d+)_(\d+)\]_logic_cycle:\s*(\d+)")
    instr_re = re.compile(r"Logic\[(\d+)_(\d+)_(\d+)\]_num_instructions:\s*(\d+)")

    max_cycles = 0
    total_instructions = 0
    with log_path.open() as f:
        for line in f:
            m = cycle_re.search(line)
            if m:
                max_cycles = max(max_cycles, int(m.group(4)))
                continue
            m = instr_re.search(line)
            if m:
                total_instructions += int(m.group(4))
    if max_cycles == 0:
        sys.exit(f"no Logic[*]_logic_cycle entries found in {log_path}")
    return max_cycles, total_instructions


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def t_transfer(bytes_total: int, alpha_key: str, beta_key: str) -> float:
    """Time for a single parallel transfer of bytes_total bytes."""
    return COEFFS[alpha_key] + bytes_total / COEFFS[beta_key]


def args_by_index(recipe: dict) -> dict:
    return {a["index"]: a for a in recipe["args"]}


def execution_input_bytes(execution: dict, args: dict, dpu_count: int) -> tuple[int, int]:
    """Return (MRAM bytes, WRAM bytes) sent to the DPUs in one execution.

    Both are summed over all DPUs: vector inputs go to MRAM, scalar inputs are
    broadcast to WRAM symbols and counted once per DPU.
    """
    mram = 0
    wram = 0
    for entry in execution["inputs"].get("mram", []):
        a = args[entry["arg_index"]]
        bytes_per_dpu = execution["elements_per_dpu"] * a["element_size_bytes"]
        mram += bytes_per_dpu * dpu_count
    # The HostData struct (8 bytes) is broadcast every round
    wram += 8 * dpu_count
    for entry in execution["inputs"].get("wram", []):
        a = args[entry["arg_index"]]
        # A broadcast still writes one copy per DPU.
        wram += a["element_size_bytes"] * dpu_count
    return mram, wram


def execution_output_bytes(execution: dict, args: dict, dpu_count: int) -> int:
    """Return the MRAM bytes copied back to the host in one execution.

    Filtered outputs are counted at their full (uncompacted) length, since
    DaPPA copies them back with holes and compacts them on the host.
    """
    out = 0
    for entry in execution["outputs"].get("mram", []):
        a = args[entry["arg_index"]]
        out += execution["elements_per_dpu"] * a["element_size_bytes"] * dpu_count
    # WRAM symbols (filter lengths, scan chunk sums) are small
    for sym in execution["outputs"].get("wram_symbols", []):
        out += 4 * 11 * dpu_count  # One uint32 per tasklet and DPU
    return out


def host_postprocess_seconds(recipe: dict, args: dict) -> dict:
    """Estimate host post-processing time, broken down per operation."""
    breakdown = {"compress_filtered_s": 0.0, "combine_reduce_s": 0.0, "combine_scan_s": 0.0}
    hp = recipe.get("host_post_processing", {})
    dpu_count = recipe["pipeline"]["dpu_count"]

    for entry in hp.get("compress_filtered", []):
        a = args[entry["arg_index"]]
        # Upper bound: every output element is moved once
        bytes_to_move = recipe["pipeline"]["data_length"] * a["element_size_bytes"]
        breakdown["compress_filtered_s"] += bytes_to_move / COEFFS["host_memmove_Bps"]

    for entry in hp.get("combine_reduce", []):
        a = args[entry["arg_index"]]
        # One combiner call per DPU and tasklet
        breakdown["combine_reduce_s"] += dpu_count * 11 * COEFFS["host_combine_op_s"]

    if any(a.get("flags") and "Scanned" in a["flags"] for a in recipe["args"]):
        # The scan stitch walks the whole output once per scanned arg
        for a in recipe["args"]:
            if a.get("flags") and "Scanned" in a["flags"]:
                bytes_walked = recipe["pipeline"]["data_length"] * a["element_size_bytes"]
                breakdown["combine_scan_s"] += bytes_walked / COEFFS["host_memmove_Bps"]

    return breakdown


def estimate(recipe: dict, dpu_cycles: int, dpu_freq_mhz: int) -> dict:
    args = args_by_index(recipe)
    dpu_count = recipe["pipeline"]["dpu_count"]
    n_exec = len(recipe["executions"])
    dpu_freq_hz = dpu_freq_mhz * 1e6

    # uPIMulator simulates all executions back to back, so the cycle count
    # covers every round. The per-round figure assumes equal rounds.
    dpu_kernel_total_s = dpu_cycles / dpu_freq_hz
    per_round_kernel_s = dpu_kernel_total_s / max(n_exec, 1)

    total_cpu_to_dpu_s = 0.0
    total_dpu_to_cpu_s = 0.0
    total_input_bytes = 0
    total_output_bytes = 0

    for execution in recipe["executions"]:
        mram_in, wram_in = execution_input_bytes(execution, args, dpu_count)
        out = execution_output_bytes(execution, args, dpu_count)
        # MRAM and WRAM inputs share the same transfer path
        t_in = t_transfer(mram_in + wram_in, "cpu_to_dpu_alpha_s", "cpu_to_dpu_beta_Bps")
        t_out = t_transfer(out, "dpu_to_cpu_alpha_s", "dpu_to_cpu_beta_Bps")
        total_cpu_to_dpu_s += t_in
        total_dpu_to_cpu_s += t_out
        total_input_bytes += mram_in + wram_in
        total_output_bytes += out

    host_post = host_postprocess_seconds(recipe, args)
    host_post_total = sum(host_post.values())

    end_to_end_s = (
        COEFFS["dpu_alloc_s"]
        + total_cpu_to_dpu_s
        + dpu_kernel_total_s
        + total_dpu_to_cpu_s
        + host_post_total
    )

    return {
        "workload": recipe.get("workload", "(unspecified)"),
        "pipeline": recipe["pipeline"],
        "num_executions": n_exec,
        "input_bytes_total": total_input_bytes,
        "output_bytes_total": total_output_bytes,
        "breakdown_s": {
            "dpu_alloc": COEFFS["dpu_alloc_s"],
            "cpu_to_dpu": total_cpu_to_dpu_s,
            "dpu_kernel": dpu_kernel_total_s,
            "dpu_to_cpu": total_dpu_to_cpu_s,
            **{f"host_{k}": v for k, v in host_post.items()},
        },
        "per_round_s": {
            "cpu_to_dpu": total_cpu_to_dpu_s / max(n_exec, 1),
            "dpu_kernel": per_round_kernel_s,
            "dpu_to_cpu": total_dpu_to_cpu_s / max(n_exec, 1),
        },
        "end_to_end_s": end_to_end_s,
        "dpu_cycles": dpu_cycles,
        "dpu_freq_mhz": dpu_freq_mhz,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_time(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s*1e3:.3f} ms"
    return f"{s*1e6:.1f} us"


def fmt_bytes(b: int) -> str:
    if b >= 1 << 30: return f"{b/(1<<30):.2f} GB"
    if b >= 1 << 20: return f"{b/(1<<20):.2f} MB"
    if b >= 1 << 10: return f"{b/(1<<10):.2f} KB"
    return f"{b} B"


def report_text(est: dict) -> str:
    out = []
    p = est["pipeline"]
    out.append(f"Workload:           {est['workload']}")
    out.append(f"Pipeline:           {p['dpu_count']} DPU x {p['num_tasklets']} tasklets,"
               f" {est['num_executions']} rounds,"
               f" {p['data_length']:,} elements x {p['element_size_bytes']} B")
    out.append(f"DPU cycles (max):   {est['dpu_cycles']:,} @ {est['dpu_freq_mhz']} MHz")
    out.append(f"Input bytes:        {fmt_bytes(est['input_bytes_total'])}")
    out.append(f"Output bytes:       {fmt_bytes(est['output_bytes_total'])}")
    out.append("")
    out.append("End-to-end breakdown:")
    for k, v in est["breakdown_s"].items():
        pct = 100 * v / est["end_to_end_s"] if est["end_to_end_s"] else 0
        out.append(f"  {k:30s} {fmt_time(v):>12s}  ({pct:5.1f}%)")
    out.append(f"  {'TOTAL':30s} {fmt_time(est['end_to_end_s']):>12s}")
    out.append("")
    out.append("Per-round average:")
    for k, v in est["per_round_s"].items():
        out.append(f"  {k:30s} {fmt_time(v):>12s}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DaPPA end-to-end cost model")
    ap.add_argument("recipe_dir", type=Path, help="directory written via DAPPA_EMIT_RECIPE")
    ap.add_argument("bin_dir", type=Path, help="uPIMulator output directory containing log.txt")
    ap.add_argument("--frequency-mhz", type=int, default=DEFAULT_DPU_FREQ_MHZ,
                    help="DPU clock frequency (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    recipe = load_recipe(args.recipe_dir)
    dpu_cycles, dpu_instructions = load_dpu_cycles(args.bin_dir)

    est = estimate(recipe, dpu_cycles, args.frequency_mhz)
    est["dpu_instructions_summed"] = dpu_instructions

    if args.json:
        print(json.dumps(est, indent=2, default=str))
    else:
        print(report_text(est))


if __name__ == "__main__":
    main()
