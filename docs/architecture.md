# DaPPA Internals

This document describes how DaPPA turns a pipeline into a UPMEM program. It is
meant for readers who want to modify or extend the framework. See the
[programming guide](programming_guide.md) for how to use it.

## Source layout

| File | Contents |
|---|---|
| `include/dappa.h` | User-facing header: primitive and argument macros, compile-time switches. |
| `include/pipeline.h`, `src/pipeline.cpp`, `src/pipeline.tpp` | `Pipeline`: stage registration, data layout, code generation, execution, recipe and bbop emission. |
| `include/pipeline_full.h`, `src/pipeline_full.cpp` | `PipelineFull`: splits non-fusible stage sequences into several `Pipeline`s. |
| `include/kernel.h`, `src/kernel.cpp` | `Kernel`: one stage, its arguments, and its per-tasklet cache layout. |
| `include/arg*.h`, `src/arg*.cpp` | `Arg`/`ArgTyped`: host buffers, flags, padding, reduction buffers. |
| `include/device.h`, `src/device.cpp` | Thin wrapper over the UPMEM host library (allocation, transfers, launch). |
| `src/device_template.c.inja` | Inja template of the DPU program. |
| `include/shared.h` | Structs shared between the host and the DPU program. |
| `include/logging.h` | Log macros for both the host and the DPU program. |

## From a pipeline to a DPU program

1. **Stage registration.** Each primitive macro expands into the arguments of
   `Pipeline::stage()`: the stage function as source text (`#func`), the same
   function as a `std::function` (used for the host-side share of the data),
   a tuple of typed arguments, and the window overlap and group size.
   `stage()` turns these into a `Kernel` and records every argument by its host
   pointer, so that later stages can refer to the outputs of earlier ones.
2. **Data layout** (`calcElementCounts`, `calcOffsets`). DaPPA splits the
   input into DPUs, tasklets, and rounds. A round is the amount of data that
   fits into the MRAM of all DPUs at once; larger inputs are processed in
   several rounds. Each tasklet streams its share through a small WRAM cache
   in blocks that keep every argument 8-byte aligned in MRAM. Windowed stages
   add an overlap region at tasklet boundaries.
3. **Code generation** (`buildJson`). The stages and the layout are rendered
   into `src/device_template.c.inja`. The template, `logging.h`, and
   `shared.h` are embedded into `libdappa` at build time with `incbin`, so the
   library has no run-time dependency on the source tree.
4. **Compilation and execution.** At `execute()` time the generated C code is
   written to a temporary directory and compiled with
   `dpu-upmem-dpurte-clang`. The host then loads the binary, and, for every
   round, copies the inputs, launches the DPUs, and copies the fetched outputs
   back. A configurable share of the elements (`DAPPA_TARGET_CPU_LOAD`) is
   processed on the host in parallel, using the `std::function` versions of the
   stages.
5. **Post-processing.** Filtered outputs are compacted on the host, partial
   reductions of all tasklets and DPUs are merged with the combine function,
   and the per-block results of scans are offset into a global scan.

## Scheduling decisions

All decisions are driven by the argument flags (`ArgFlags`), not by the C++
types:

- **Vectors** (`INPUT`, `OUTPUT`, `INOUT`) are sharded across DPUs and
  tasklets. **Scalars** (`PARAMETER`) are broadcast and stored once in WRAM.
  **Lookup tables** (`LOOKUP_INPUT`) are broadcast once into the MRAM of every
  DPU and read on demand.
- **Fusion.** All stages of a `Pipeline` run in one DPU program, one element
  block at a time, so that intermediate vectors never leave WRAM. A stage that
  reads a filtered or reduced vector cannot be fused; `PipelineFull` starts a
  new `Pipeline` at that point.
- **Filtering.** Filter stages mark elements as invalid instead of compacting
  them on the DPU. Compaction happens on the host while the next round runs,
  which avoids the synchronization between tasklets that on-DPU compaction
  requires.
- **Reductions.** Each reduction accumulator is replicated R times in WRAM.
  With R equal to the number of tasklets, every tasklet updates a private copy
  without locking. Large accumulators (e.g., histograms with many bins) use
  fewer copies, each protected by a mutex, so that enough WRAM remains for the
  input cache. If the accumulator does not fit in WRAM at all, the pipeline
  runs on the host.
- **Interleaved indexing.** Pipelines consisting only of map and reduce stages
  assign blocks to tasklets in a strided order, so that the tasklets' MRAM
  accesses are spread over time instead of contending in lockstep.
- **Scans** are computed in two passes per DPU (a local scan per tasklet,
  followed by applying the prefix of the preceding tasklets). The host then
  offsets each (round, DPU) block by the fold of all preceding blocks.

## DPU set caching

`dpu_alloc` is the most expensive host operation. The DPU set is therefore
allocated by the first `Pipeline` and reused by all later ones that use the
same number of DPUs (e.g., the layers of an MLP), which only load their own
program. The set is released at process exit.

## Device parameters

`include/device.h` contains the parameters of the target DPU: 11 tasklets
(enough to fill the 11-stage revolver pipeline), 64 KB of WRAM, 64 MB of
MRAM, 512-byte tasklet stacks, and the MRAM and WRAM alignment. `backend`
selects the UPMEM backend (`"simulator"` for the SDK's functional simulator,
`""` for real hardware).

## Offline backends

`execute()` has two additional modes that stop before compiling the DPU
program:

- `DAPPA_EMIT_RECIPE=<dir>` (`emitRecipe`) writes the generated code together
  with the exact MRAM and WRAM contents of every DPU for every round. See
  [`integrations/upimulator`](../integrations/upimulator/README.md).
- `DAPPA_EMIT_BBOP=<dir>` (`emitBbop`) lowers the stages to bulk bitwise
  operations for processing-using-DRAM. It runs before the UPMEM data layout
  is computed, since the backend does its own tiling. See
  [`integrations/simdram`](../integrations/simdram/README.md).
