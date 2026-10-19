#ifndef DAPPA_KERNEL_ARG_H
#define DAPPA_KERNEL_ARG_H

#include "arg.h"

namespace DaPPA
{

class KernelArg : public ArgFlagsBase
{
public:
    KernelArg();
    KernelArg(Arg *_arg, const ArgFlagsBase &_flags, uint32_t overlap, const void *paddingPtr = nullptr, bool isPaddingPtr = true);
    nlohmann::json toJson();

    Arg *base;
    uint32_t cacheOffset;
    uint32_t inputOverlap;
    std::vector<uint8_t> paddingBuffer;

private:
    nlohmann::json json;
};

}

#endif //DAPPA_KERNEL_ARG_H
