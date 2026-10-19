// Element-wise vector addition: c[i] = a[i] + b[i].

#include <cstdio>
#include <vector>
#include "dappa.h"
#include "common.h"

int main()
{
    const uint64_t length = 1 << 20;
    std::vector<int32_t> a(length), b(length), c(length);
    for (uint64_t i = 0; i < length; i++)
    {
        a[i] = static_cast<int32_t>(i);
        b[i] = static_cast<int32_t>(2 * i);
    }

    DaPPA::Pipeline p(length);
    p.stage(MAP(([](int32_t *a, int32_t *b, int32_t *c){
                    *c = *a + *b;
                }),
                INPUT(int32_t, a),
                INPUT(int32_t, b),
                OUTPUT(int32_t, c)));
    p.fetch(c);
    if (!p.execute())
        return 1;
    if (emitOnly())
        return 0;

    for (uint64_t i = 0; i < length; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            std::printf("Mismatch at %lu: %d != %d\n", i, c[i], a[i] + b[i]);
            return 1;
        }
    }
    std::printf("vector_add: OK (%lu elements)\n", length);
    return 0;
}
