// Dot product as a fused MAP -> REDUCE pipeline. The intermediate vector c is
// never fetched, so it stays on the DPUs and needs no host buffer.

#include <cstdio>
#include <vector>
#include "dappa.h"
#include "common.h"

int main()
{
    const uint64_t length = 1 << 20;
    std::vector<uint32_t> a(length), b(length);
    for (uint64_t i = 0; i < length; i++)
    {
        a[i] = i % 16;
        b[i] = i % 7;
    }
    uint32_t c;
    uint32_t sum = 0;

    DaPPA::Pipeline p(length);
    p.stage(MAP(([](uint32_t *c, uint32_t *a, uint32_t *b){
                    *c = *a * *b;
                }),
                OUTPUT(uint32_t, &c),
                INPUT(uint32_t, a),
                INPUT(uint32_t, b)));
    p.stage(REDUCE(([](uint32_t *sum, uint32_t *c){
                       *sum += *c;
                   }),
                   REDUCE_OUT(uint32_t, &sum),
                   INPUT(uint32_t, &c)));
    p.fetch(&sum);
    if (!p.execute())
        return 1;
    if (emitOnly())
        return 0;

    uint32_t expected = 0;
    for (uint64_t i = 0; i < length; i++)
        expected += a[i] * b[i];

    std::printf("dot_product: %u (expected %u) %s\n", sum, expected, sum == expected ? "OK" : "FAILED");
    return sum == expected ? 0 : 1;
}
