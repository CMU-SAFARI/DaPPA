// Keep only the even elements of a vector (in place). The result length is
// only known after execution and is queried with getLength().

#include <cstdio>
#include <vector>
#include "dappa.h"
#include "common.h"

int main()
{
    const uint64_t length = 1 << 20;
    std::vector<uint64_t> x(length);
    for (uint64_t i = 0; i < length; i++)
        x[i] = (i * 2654435761u) % 1000;

    std::vector<uint64_t> expected;
    for (auto v : x)
        if (v % 2 == 0)
            expected.push_back(v);

    DaPPA::Pipeline p(length);
    p.stage(FILTER(([](uint64_t *x){
                       return *x % 2 == 0;
                   }),
                   INOUT(uint64_t, x)));
    p.fetch(x);
    if (!p.execute())
        return 1;
    if (emitOnly())
        return 0;

    uint64_t resultLength = p.getLength(x);
    bool ok = resultLength == expected.size();
    for (uint64_t i = 0; ok && i < resultLength; i++)
        ok = x[i] == expected[i];

    std::printf("select: kept %lu of %lu elements %s\n", resultLength, length, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
