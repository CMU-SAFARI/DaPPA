# DaPPA: A Unified Data-Parallel Programming Framework for Efficient Processing-in-Memory Execution

DaPPA (**da**ta-**p**arallel **p**rocessing-in-memory **a**rchitecture) is a
programming framework for general-purpose processing-in-memory (PIM) systems.
Programmers express computation once, with high-level data-parallel patterns
(map, filter, reduce, scan, window, group) connected in a dataflow pipeline.
DaPPA then automatically generates all substrate-specific data organization:

- **Processing-near-memory (UPMEM).** DaPPA generates and compiles the DPU
  program at run time, partitions the data across DPUs and tasklets, manages
  MRAM/WRAM allocation and host-DPU transfers, and fuses consecutive stages
  into one kernel.
- **Processing-using-memory (SIMDRAM).** The same pipeline is lowered to bulk
  bitwise operations with multi-level tiling across the SIMD lanes of DRAM
  rows.

```cpp
#include "dappa.h"

DaPPA::Pipeline p(n);
p.stage(MAP(([](uint32_t *c, uint32_t *a, uint32_t *b){ *c = *a * *b; }),
            OUTPUT(uint32_t, &c), INPUT(uint32_t, a), INPUT(uint32_t, b)));
p.stage(REDUCE(([](uint32_t *sum, uint32_t *c){ *sum += *c; }),
               REDUCE_OUT(uint32_t, &sum), INPUT(uint32_t, &c)));
p.fetch(&sum);
p.execute();    // dot product of a and b on the DPUs
```

## Contents

- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Documentation](#documentation)
- [Simulation and other PIM substrates](#simulation-and-other-pim-substrates)
- [Citation](#citation)
- [License](#license)

## Getting started

### Requirements

- CMake 3.13+ and a C++17 compiler (GCC 9+ or Clang 10+)
- The [UPMEM SDK](https://github.com/CMU-SAFARI/UPMEM) (tested with 2023.1.0)

DaPPA runs on real UPMEM systems and on the functional simulator included in
the SDK, which is the default (see [Configuration](docs/programming_guide.md#configuration)).

### Build

With the SDK installed and `UPMEM_HOME` pointing to it:

```sh
source $UPMEM_HOME/upmem_env.sh
cmake --preset release            # or: cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build --preset release -j
```

Alternatively, use the provided Docker image, which includes the SDK:

```sh
docker build -t dappa -f docker/Dockerfile .
docker run --rm -it -v "$PWD":/dappa -w /dappa dappa
```

### Run

```sh
./build/release/examples/vector_add
./build/release/examples/dot_product
./build/release/tests/tests                 # full test suite (Catch2)
./build/release/tests/tests "[workload]"    # only the workloads
```

The DPU code generator (`dpu-upmem-dpurte-clang`) must be in `PATH` at run
time, since DaPPA compiles the DPU program when the pipeline executes.

## Repository structure

```
.
├── include/            Public headers (dappa.h is the only one users include)
├── src/                Host library and the DPU code template
├── examples/           Small standalone programs
├── tests/              Unit tests and workloads (Catch2)
├── docs/               Programming guide and internals
├── integrations/
│   ├── upimulator/     Cycle-accurate simulation with uPIMulator and end-to-end cost model
│   └── simdram/        Processing-using-DRAM backend driver
├── scripts/            Measurement helper
├── docker/             Build environment with the UPMEM SDK
└── third_party/        Catch2, Inja, nlohmann/json, incbin
```

The workloads in `tests/workloads.cpp` include vector addition, GEMV,
selection, unique, reduction, prefix sum, small and large histograms, binary
search, time-series analysis, linear and logistic regression, k-means, and a
three-layer MLP. Each one checks its result against a CPU reference.

## Documentation

- [Programming guide](docs/programming_guide.md): primitives, arguments,
  rules for stage functions, and configuration.
- [Internals](docs/architecture.md): code generation, data layout, and
  scheduling.

## Simulation and other PIM substrates

- [`integrations/upimulator`](integrations/upimulator/README.md): run any
  DaPPA pipeline in the cycle-accurate
  [uPIMulator](https://github.com/VIA-Research/uPIMulator) simulator and
  estimate its end-to-end execution time, without UPMEM hardware.
- [`integrations/simdram`](integrations/simdram/README.md): lower DaPPA
  pipelines to a SIMDRAM-like processing-using-DRAM substrate and evaluate
  them with an analytical cost model.

## Citation

If you use DaPPA, please cite:

```bibtex
@inproceedings{oliveira2026dappa,
    author = "Oliveira, Geraldo F and Kohli, Alain and Novo, David and G{\'o}mez-Luna, Juan and Mutlu, Onur",
    title = "{DaPPA: A Unified Data-Parallel Programming Framework for Efficient Processing-in-Memory Execution}",
    booktitle = "ICCD",
    year = "2026"
}
```

## Acknowledgments

DaPPA builds on the UPMEM programming framework developed in Alain Kohli's
Master's thesis at ETH Zürich ("Framework for Easy Use of the UPMEM
Processing-in-Memory Platform using Parallel Patterns", 2023). We thank the
authors of [PrIM](https://github.com/CMU-SAFARI/prim-benchmarks),
[uPIMulator](https://github.com/VIA-Research/uPIMulator), and
[Proteus](https://github.com/CMU-SAFARI/Proteus), on which our evaluation
infrastructure relies.

## License

DaPPA is released under the [MIT License](LICENSE). The libraries in
`third_party/` are distributed under their own licenses (see
[`third_party/README.md`](third_party/README.md)).

Questions and bug reports are welcome as GitHub issues.
