# Simulating DaPPA pipelines with uPIMulator

[uPIMulator](https://github.com/VIA-Research/uPIMulator) is a cycle-accurate
simulator of the UPMEM PIM architecture. This directory lets you run any DaPPA
pipeline in uPIMulator (Go version) without UPMEM hardware.

The flow has three steps:

1. **Emit a recipe.** Running a DaPPA program with `DAPPA_EMIT_RECIPE=<dir>`
   makes `Pipeline::execute()` write the generated DPU program and all of its
   inputs to `<dir>` instead of compiling and launching it.
2. **Simulate.** A generic `DAPPA` benchmark in uPIMulator compiles the
   generated DPU program and loads the recipe inputs into the simulated DPUs.
3. **Estimate end-to-end time.** `cost_model.py` combines the simulated DPU
   cycles with an analytical model of DPU allocation, CPU-DPU transfers, and
   host post-processing.

## Setup

Requirements: Go, Python 3, and Docker (uPIMulator compiles DPU code inside a
Docker container, so your user must be in the `docker` group).

```sh
./setup.sh            # clones uPIMulator into ./uPIMulator, patches, and builds it
```

`setup.sh` checks out the tested uPIMulator commit, applies
`upimulator.patch`, and copies the files under `files/` into the uPIMulator
tree. The patch:

- registers the `DAPPA` benchmark (`src/assembler/prim/dappa.go`,
  `benchmark/DAPPA/`), which reads the recipe from `$DAPPA_RECIPE_DIR`;
- adds `DAPPA_SKIP_VERIFY`, which disables uPIMulator's output check (recipes
  carry no reference outputs);
- implements the `S_RRR`/`U_RRR` and `S_ERRI`/`U_ERRI` instruction variants,
  which upstream leaves unimplemented and which DaPPA-generated code can use;
- installs the UPMEM SDK in uPIMulator's Docker image from the
  [SAFARI archive](https://github.com/CMU-SAFARI/UPMEM), since the original
  download location is offline.

## Usage

```sh
# 1. Emit a recipe from any DaPPA program (examples, tests, or your own code).
DAPPA_EMIT_RECIPE=/tmp/va ./build/release/examples/vector_add
DAPPA_EMIT_RECIPE=/tmp/sel ./build/release/tests/tests "SEL: Selection"

# 2. + 3. Simulate it and print the end-to-end estimate.
integrations/upimulator/run.sh /tmp/va
```

`run.sh` reads the DPU and tasklet counts from `recipe.json`, places the
generated `device.c` into the `DAPPA` benchmark, runs uPIMulator, and calls
the cost model. The simulator statistics are written to `<recipe>/bin/log.txt`.
The kernel time of a pipeline is the maximum `logic_cycle` over all DPUs.

When a test binary is used to emit a recipe, its result check fails because no
results are computed. This is expected.

## Recipe format

```
<dir>/
  recipe.json                    # pipeline shape, per-round transfers, MRAM offsets
  device.c                       # generated DPU program
  inputs/exec<r>/dpu<d>/arg<i>.bin   # MRAM contents of every DPU for every round
```

`recipe.json` (`"$schema_version": 1`) lists, for every execution round, the
vector inputs to copy into each DPU's MRAM heap, the scalar inputs to write to
WRAM symbols, and the outputs to read back. The consumer is
`files/src/assembler/prim/dappa.go`.

## Files

| File | Description |
|---|---|
| `setup.sh` | Clone, patch, and build uPIMulator. |
| `run.sh` | Simulate a recipe and run the cost model. |
| `cost_model.py` | End-to-end time estimate from a recipe and a uPIMulator log. |
| `upimulator.patch` | Changes to existing uPIMulator sources. |
| `files/` | New files added to the uPIMulator tree (`DAPPA` benchmark and recipe loader). |
