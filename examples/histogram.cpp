// 256-bin histogram of 12-bit values. The reduction output is an array, so
// REDUCE_OUT takes its size in bytes and a host-side function that merges
// the partial histograms of different tasklets and DPUs.

#include <cstdio>
#include <vector>
#include "dappa.h"
#include "common.h"

int main()
{
    const uint64_t length = 1 << 20;
    const uint32_t bins = 256;
    std::vector<uint32_t> pixels(length);
    std::vector<uint32_t> hist(bins, 0);
    for (uint64_t i = 0; i < length; i++)
        pixels[i] = (i * 2654435761u) % 4096;

    DaPPA::Pipeline p(length);
    p.stage(REDUCE(([](uint32_t *hist, uint32_t *pixel){
                       hist[(*pixel * 256) >> 12]++;
                   }),
                   REDUCE_OUT(uint32_t, hist.data(), ([](uint32_t *out, const uint32_t *in){
                       for (uint32_t i = 0; i < 256; i++)
                           out[i] += in[i];
                   }), bins * sizeof(uint32_t)),
                   INPUT(uint32_t, pixels)));
    p.fetch(hist);
    if (!p.execute())
        return 1;
    if (emitOnly())
        return 0;

    std::vector<uint32_t> expected(bins, 0);
    for (auto v : pixels)
        expected[(v * bins) >> 12]++;

    bool ok = hist == expected;
    std::printf("histogram: %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
