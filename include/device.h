#ifndef DAPPA_DEVICE_H
#define DAPPA_DEVICE_H

#include <cstdint>
#include <string>
#include "arg.h"

extern "C" {
    #include <dpu/dpu.h>
}

namespace DaPPA
{

class Device
{
public:
    // System parameters
    static constexpr char backend[] = "simulator";
    static uint32_t dpuCountAvailable;
    // MRAM size of a real UPMEM DPU (64 MB). Smaller values force more rounds.
    static constexpr int32_t mramSize = 1024 * 65536;
    static constexpr int8_t mramAlignment = 8;
    static constexpr int8_t wramAlignment = 4;
    static constexpr uint32_t taskletStackSize = 512;
    static constexpr uint32_t taskletCountPerDpu = 11;
    static constexpr uint32_t wramSize = 65536;
    static constexpr uint32_t wramBuffer = 256;
    static constexpr uint32_t maxWramTransferSize = 2048;

    static constexpr uint32_t paddedSizeMram(uint32_t size)
    {
        return ((size + mramAlignment - 1) / mramAlignment) * mramAlignment;
    }

    static constexpr uint32_t paddedSizeWram(uint32_t size)
    {
        return ((size + wramAlignment - 1) / wramAlignment) * wramAlignment;
    }

    Device(const std::string &programFile, uint32_t _dpuCountUsed);
    ~Device();

    // The DPU set is allocated once per process and reused by later Pipelines that use the
    // same number of DPUs, which then only need to load their program. Freed at exit.
    static dpu_set_t cachedSet;
    static uint32_t  cachedDpuCount;
    static bool      hasCachedSet;

    bool launch();
    bool readLog();

    bool transferToDpus(Arg &arg, uint64_t baseIndex, uint64_t elementsPerDpu);
    bool broadcastToDpus(Arg &arg);
    bool broadcastVectorToDpus(Arg &arg);

    template<typename T>
    bool broadcastToDpus(const std::string &symbol, const T &value)
    {
        DPU_CHECK(dpu_broadcast_to(this->set, symbol.c_str(), 0, &value, sizeof(value), DPU_XFER_DEFAULT), {return false;});
        return true;
    }

    bool transferFromDpus(Arg &arg, uint64_t baseIndex, uint64_t elementsPerDpu);
    bool transferFromDpus(Arg &arg);
    bool transferFromDpus(const std::string &symbol, std::vector<std::vector<uint32_t> > &dest);
    // Reads a __host scalar of elementSize bytes from each DPU into dest[dpu]
    bool transferScalarBytesFromDpus(const std::string &symbol, uint32_t elementSize,
                                     std::vector<std::vector<uint8_t> > &dest);

    static void detectDpus();

private:
    dpu_set_t set;
    uint32_t dpuCountUsed;
};

}

#endif //DAPPA_DEVICE_H
