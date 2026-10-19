# Processing-using-DRAM backend (SIMDRAM)

Besides UPMEM, DaPPA can lower a pipeline to SIMDRAM, a
processing-using-DRAM (PuD) substrate that executes bit-serial operations
across the 65,536 bitlines of a DRAM row. The same front-end code is used: no
changes to the pipeline are needed.

Setting `DAPPA_EMIT_BBOP=<dir>` makes `Pipeline::execute()` write
`<dir>/harness.c` instead of running on UPMEM. The harness expresses the
pipeline as a sequence of bulk bitwise operations (bbops) for the SIMDRAM
analytical cost model:

| Stage | Lowering |
|---|---|
| `MAP` | One element-wise bbop. |
| `REDUCE` | One bbop reduction. |
| `GROUP` | N independent reductions of length G, mapped either *output-stationary* (the N outputs are packed across the SIMD lanes and the reduction becomes G sequential multiply-accumulate steps) or *reduction-in-lane* (one vectorized length-G reduction per output). The backend picks the mapping with fewer multiplication passes. |

Output-stationary mapping lets DaPPA outperform compilers that only vectorize
the innermost loop on nested-loop kernels such as GEMM. On GEMV-, reduction-,
and stencil-shaped kernels both mappings are the same.
Operands of `GROUP` stages are assumed to be square matrices.

## Usage

The harness is compiled against the bbop cost model (`bbop_manager.c` and
`bbop_manager.h`):

```sh
export BBOP_UTIL_DIR=/path/to/bbop/util
DAPPA_PUD_NOUT=1024 DAPPA_PUD_G=1024 \
    integrations/simdram/run.sh /tmp/pud -- build/release/tests/tests "group: PUD parametric"
```

`run.sh` runs the program with `DAPPA_EMIT_BBOP`, builds and runs the harness,
and prints the `SIMDRAM_64` compute latency in milliseconds from
`bbop_statistics.csv`.

The `[pud]` tests in `tests/pud.cpp` emit harnesses for a square matrix
multiplication (`DAPPA_2MM_N`) and for a generic `GROUP` stage with `NOUT`
outputs of contraction length `G` (`DAPPA_PUD_NOUT`, `DAPPA_PUD_G`). They do
nothing unless `DAPPA_EMIT_BBOP` is set.

## Cost model

The harness uses the bbop API of the
[Proteus](https://github.com/CMU-SAFARI/Proteus) PuD framework
(`util/bbop_manager.{c,h}`), extended with `bbop_register_input()` and
`bbop_register_output()` to account for the horizontal-to-vertical data
transposition, and with a `BBOP_SKIP_COMPUTE` option that only accounts for
latency without computing results. The harness never touches real data, so
host buffers in the DaPPA program can be dummies.
