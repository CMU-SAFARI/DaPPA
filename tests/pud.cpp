#include <cstdlib>
#include "catch.hpp"
#include "common.h"

using namespace DaPPA;

// Square matrix multiplication C = A * B as a single GROUP stage (n * n outputs,
// each a length-n dot product). With DAPPA_EMIT_BBOP set, the pipeline emits a
// bbop harness for the SIMDRAM cost model instead of running on UPMEM. Only the
// stage and argument shapes matter, so the host buffers are dummies.
// n is read from DAPPA_2MM_N (default 32000, the PolyBench 2mm size).
TEST_CASE("matmul: PUD output-stationary", "[pud]")
{
    if (std::getenv("DAPPA_EMIT_BBOP") == nullptr)
        return;

    uint64_t n = 32000;
    if (const char *e = std::getenv("DAPPA_2MM_N"))
        n = std::strtoull(e, nullptr, 10);
    uint32_t cols = static_cast<uint32_t>(n);

    std::vector<int32_t> A(1), B(1), C(1);  // Never dereferenced

    Pipeline p(n * n * n);

    p.stage(GROUP([](int32_t *A, int32_t *B, int32_t *C, uint32_t *cols){
                      int32_t res = 0;
                      for (uint32_t k = 0; k < *cols; k++)
                          res += A[k] * B[k];
                      *C = res;
                  },
                  cols,
                  INPUT(int32_t, A),
                  INPUT(int32_t, B),
                  OUTPUT(int32_t, C),
                  PARAMETER(uint32_t, &cols)));

    p.fetch(C);
    REQUIRE(p.execute());
}

// A single GROUP stage with nOut outputs (DAPPA_PUD_NOUT), each reducing g
// inputs (DAPPA_PUD_G). Covers any GEMM- or GEMV-shaped kernel, so the PUD
// backend's choice between output-stationary and reduction-in-lane mapping can
// be checked for arbitrary shapes. Runs only with DAPPA_EMIT_BBOP set.
TEST_CASE("group: PUD parametric", "[pud]")
{
    if (std::getenv("DAPPA_EMIT_BBOP") == nullptr)
        return;

    uint64_t nOut = 1024;
    uint64_t g = 1024;
    if (const char *e = std::getenv("DAPPA_PUD_NOUT"))
        nOut = std::strtoull(e, nullptr, 10);
    if (const char *e = std::getenv("DAPPA_PUD_G"))
        g = std::strtoull(e, nullptr, 10);
    uint32_t cols = static_cast<uint32_t>(g);

    std::vector<int32_t> A(1), B(1), C(1);  // Never dereferenced

    Pipeline p(nOut * g);

    p.stage(GROUP([](int32_t *A, int32_t *B, int32_t *C, uint32_t *cols){
                      int32_t res = 0;
                      for (uint32_t k = 0; k < *cols; k++)
                          res += A[k] * B[k];
                      *C = res;
                  },
                  cols,
                  INPUT(int32_t, A),
                  INPUT(int32_t, B),
                  OUTPUT(int32_t, C),
                  PARAMETER(uint32_t, &cols)));

    p.fetch(C);
    REQUIRE(p.execute());
}
