# DaPPA Programming Guide

A DaPPA program describes a computation as a *pipeline* of data-parallel
stages. Each stage applies a user function to the elements of one or more
vectors. DaPPA then generates the DPU code, distributes the data across DPUs
and tasklets, runs the pipeline, and gathers the results.

```cpp
#include "dappa.h"

DaPPA::Pipeline p(length);      // number of elements the pipeline processes
p.stage(...);                   // add one or more stages
p.fetch(result);                // select the outputs to copy back to the host
p.execute();                    // generate, compile, and run the DPU program
```

## Contents

- [Stages](#stages)
- [Arguments](#arguments)
- [Primitives](#primitives)
- [Fetching results and executing](#fetching-results-and-executing)
- [Pipelines that cannot be fused](#pipelines-that-cannot-be-fused)
- [Rules for stage functions](#rules-for-stage-functions)
- [Configuration](#configuration)

## Stages

A stage is added with a primitive macro:

```cpp
p.stage(PRIMITIVE((FUNC),
                  [STAGE_PARAMETER,]
                  IO_1(TYPE_1, PTR_1),
                  ...
                  IO_N(TYPE_N, PTR_N)));
```

| Field | Meaning |
|---|---|
| `PRIMITIVE` | One of the primitives listed below (`MAP`, `FILTER`, `REDUCE`, ...). |
| `FUNC` | Function called for every element (or window, or group). A parenthesized lambda or the name of a function. |
| `STAGE_PARAMETER` | Window size and/or group size, only for primitives that take one. |
| `IO_n` | Role of argument `n` of `FUNC` (`INPUT`, `OUTPUT`, ...). |
| `TYPE_n` | Non-pointer element type of argument `n`. |
| `PTR_n` | Host pointer (or `std::vector`) holding the data of argument `n`. |

Example:

```cpp
p.stage(MAP(([](int *c, int *a, int *b){
                *c = *a + *b;
            }),
            OUTPUT(int, c),
            INPUT(int, a),
            INPUT(int, b)));
```

## Arguments

| Macro | Kind | Description |
|---|---|---|
| `INPUT(T, ptr[, padding])` | vector | Read-only input, sharded across DPUs and tasklets. `padding` supplies the values read past the end of the data by a `WINDOW` stage. |
| `OUTPUT(T, ptr)` | vector | Output, one element per input element (or per group). |
| `INOUT(T, ptr)` | vector | Read and written in place. |
| `PARAMETER(T, ptr[, sizeBytes])` | scalar | Broadcast to all DPUs and kept in WRAM. Pass `sizeBytes` for small arrays (e.g., a weight vector). |
| `REDUCE_OUT(T, ptr[, combine, sizeBytes])` | scalar | Accumulator of a `REDUCE` stage. `combine(T *out, const T *in)` merges the partial results of different tasklets and DPUs on the host. It can be omitted if the reduce function takes exactly two arguments `(acc, x)` of the same type, in which case the reduce function itself is used. Pass `sizeBytes` for array accumulators such as histograms. |
| `LOOKUP_INPUT(T, ptr, sizeBytes)` | lookup table | Read-only array stored once in the MRAM of every DPU and randomly indexed with `LOOKUP(arr, idx)`. Declare the function parameter as `LOOKUP_PTR(T)`. |

Intermediate vectors that are never fetched do not need a host buffer: pass
the address of a single dummy element, as in `examples/dot_product.cpp`.

## Primitives

| Macro | Stage parameter | Function signature | Semantics |
|---|---|---|---|
| `MAP` | | `void f(T1 *, ..., Tn *)` | One output per input element. |
| `FILTER` | | `bool f(T *x, ...)` | Keeps the elements for which `f` returns `true`. The first argument is the filtered vector (`INOUT`, or `OUTPUT` followed by the inputs). |
| `REDUCE` | | `void f(Acc *acc, T *x, ...)` | Folds all elements into a `REDUCE_OUT` accumulator. |
| `SCAN` | | `void f(T *acc, T *x)` | Inclusive prefix fold: `out[i] = out[i-1] (+) in[i]`. Arguments: `INOUT(T, out)` initialized with the identity element, then `INPUT(T, in)`. |
| `WINDOW` | `windowSize` | `void f(T *out, T *window, ...)` | Sliding window: `window[0..windowSize-1]` are consecutive input elements. |
| `GROUP` | `groupSize` | `void f(T *out, T *group, ...)` | Non-overlapping groups of `groupSize` elements, one output per group. |
| `WINDOW_FILTER` | `windowSize` | `bool f(...)` | `WINDOW` whose outputs are filtered. |
| `GROUP_FILTER` | `groupSize` | `bool f(...)` | `GROUP` whose outputs are filtered. |
| `WINDOW_GROUP` | `groupSize, windowSize` | `void f(...)` | Windows of `windowSize` elements, advancing by `groupSize`. |
| `WINDOW_GROUP_FILTER` | `groupSize, windowSize` | `bool f(...)` | `WINDOW_GROUP` whose outputs are filtered. |

See `tests/workloads.cpp` for complete examples of each primitive (vector
addition, GEMV, selection, unique, reduction, prefix sum, histograms, binary
search, time series, linear and logistic regression, k-means, and an MLP).

## Fetching results and executing

```cpp
p.fetch(out);                       // copy `out` back after execution
bool ok = p.execute();              // false if the pipeline cannot run
uint64_t n = p.getLength(out);      // number of valid elements (after filtering)
```

Only fetched arguments are copied back to the host. Vectors that are produced
and consumed on the DPUs stay in MRAM.

## Pipelines that cannot be fused

`Pipeline` fuses all stages into a single DPU program. This is not possible
when a stage reads a vector that an earlier stage filtered or reduced (e.g.,
`FILTER -> MAP` or `REDUCE -> MAP`), because the data has to be compacted or
combined on the host first. Use `PipelineFull` in that case. It has the same
interface and splits the stages into several `Pipeline`s that run one after
another:

```cpp
DaPPA::PipelineFull p(length);
p.stage(REDUCE(...));
p.stage(MAP(...));      // reads the reduced value as a PARAMETER
```

## Rules for stage functions

These constraints are only checked when the DPU code is compiled at run time:

- The function must compile both as C and as C++. Lambdas are converted to
  plain C functions in the generated DPU code.
- Lambdas must not capture anything and must be enclosed in parentheses.
- All parameters are pointers. `TYPE_n` is the non-pointer element type and
  must match the constness of the data.
- `TYPE_n` must be a primitive type or a C struct laid out according to
  Clang's alignment and padding rules. Sizes of 2, 4, or a multiple of 8 bytes
  give the best MRAM utilization.
- The I/O macros must appear in the same order as the parameters of the
  function.
- Named functions, structs, and helper code used by stage functions must be
  passed as source text with `p.addHeader(source)`, so that they are also
  compiled into the DPU program (see `tests/structs.h` and its use in
  `tests/workloads.cpp`).

## Configuration

Compile-time switches, edited in the source:

| Switch | File | Default | Description |
|---|---|---|---|
| `DAPPA_MAX_DPU_COUNT` | `include/dappa.h` | 16 | Maximum number of DPUs to allocate (0 = all available). |
| `DAPPA_TARGET_CPU_LOAD` | `include/dappa.h` | 0 | Fraction of the elements processed on the host CPU instead (0 to 1). |
| `DAPPA_SAFE_CACHE_ALLOCATION` | `include/dappa.h` | off | Allow a stage to read its inputs after writing an output (slower). |
| `VERBOSITY` | `include/logging.h` | `VERBOSITY_QUIET` | Host and DPU log level. |
| `MEASURE_PERFORMANCE_HOST`, `MEASURE_PERFORMANCE_DPU` | `include/logging.h` | off | Print per-phase host timings and per-stage DPU cycle counts. |
| `Device::backend` | `include/device.h` | `"simulator"` | UPMEM backend passed to `dpu_alloc` (`"simulator"` for the SDK functional simulator, `""` for real hardware). |

Run-time switches (environment variables):

| Variable | Description |
|---|---|
| `DAPPA_EMIT_RECIPE=<dir>` | Write the generated DPU code and all DPU inputs to `<dir>` instead of running, for simulation in uPIMulator. See [`integrations/upimulator`](../integrations/upimulator/README.md). |
| `DAPPA_EMIT_BBOP=<dir>` | Lower the pipeline to a SIMDRAM bbop harness in `<dir>` instead of running. See [`integrations/simdram`](../integrations/simdram/README.md). |
