#include "device.h"
#include <iostream>
#include "logging.h"
#include "dappa.h"

#define CHECK(x) DPU_CHECK(x, {return false;})

namespace DaPPA
{

uint32_t Device::dpuCountAvailable = DAPPA_MAX_DPU_COUNT;

// Process-wide DPU set cache
dpu_set_t Device::cachedSet;
uint32_t  Device::cachedDpuCount = 0;
bool      Device::hasCachedSet   = false;

namespace {
    // Releases the cached DPU set at process exit
    struct CachedSetReleaser {
        ~CachedSetReleaser() {
            if (Device::hasCachedSet) {
                dpu_free(Device::cachedSet);
                Device::hasCachedSet = false;
            }
        }
    };
    CachedSetReleaser cachedSetReleaser;
}

Device::Device(const std::string &programFile, uint32_t _dpuCountUsed)
        :
        dpuCountUsed(_dpuCountUsed)
{
    if (Device::hasCachedSet && Device::cachedDpuCount == _dpuCountUsed)
    {
        // Reuse the cached set, only the program needs to be loaded
        this->set = Device::cachedSet;
    }
    else
    {
        // Replace the cached set with a newly allocated one
        if (Device::hasCachedSet)
        {
            dpu_free(Device::cachedSet);
            Device::hasCachedSet = false;
        }
        DPU_ASSERT(dpu_alloc(this->dpuCountUsed, (std::string("backend=") + Device::backend).c_str(), &this->set));
        Device::cachedSet      = this->set;
        Device::cachedDpuCount = _dpuCountUsed;
        Device::hasCachedSet   = true;
    }
    DPU_ASSERT(dpu_load(this->set, programFile.c_str(), nullptr));
}

Device::~Device()
{
    // The DPU set is cached for later Pipelines and freed by CachedSetReleaser at exit
}

bool Device::launch()
{
    CHECK(dpu_launch(this->set, DPU_SYNCHRONOUS));
    return true;
}

bool Device::readLog()
{
    dpu_set_t dpu;

    DPU_FOREACH(this->set, dpu) {
        CHECK(dpu_log_read(dpu, stdout));
    }
    std::cout << std::flush;
    return true;
}

bool Device::transferToDpus(Arg &arg, uint64_t baseIndex, uint64_t elementsPerDpu)
{
    if (!arg.isVector())
        return false;

    dpu_set_t dpu;
    uint32_t n;

    DPU_FOREACH(this->set, dpu, n)
    {
        DEBUG_HOST("Input Transfer: %p[%lu] -> M+%lu [DPU %d, Size %d, Transfer Overlap %d]",
                   arg.hostPtr,
                   arg.elementSize * (baseIndex + n * elementsPerDpu) / arg.startLengthFraction,
                   arg.mramOffset,
                   n,
                   Device::paddedSizeMram(arg.elementSize * (elementsPerDpu / arg.startLengthFraction + arg.transferOverlap)),
                   arg.transferOverlap);

        CHECK(dpu_prepare_xfer(dpu, arg.hostPtr + arg.elementSize * (baseIndex + n * elementsPerDpu) / arg.startLengthFraction));
    }
    CHECK(dpu_push_xfer(this->set,
                        DPU_XFER_TO_DPU,
                        DPU_MRAM_HEAP_POINTER_NAME,
                        arg.mramOffset,
                        Device::paddedSizeMram(arg.elementSize * (elementsPerDpu / arg.startLengthFraction + arg.transferOverlap)),
                        DPU_XFER_DEFAULT));

    return true;
}

bool Device::broadcastToDpus(Arg &arg)
{
    if (!arg.isScalar())
        return false;

    dpu_set_t dpu;
    uint32_t n;
    std::string symbol = "argScalarRaw" + std::to_string(arg.index);

    // Reduce accumulators occupy R banks, read-only scalars are stored only once
    uint32_t bankCount = arg.isOutput() ? arg.replicationFactor : 1;

    for (uint32_t i = 0; i < bankCount; i++)
    {
        if (arg.isOutput())
        {
            DPU_FOREACH(this->set, dpu, n)
            {
                CHECK(dpu_prepare_xfer(dpu, arg.reduceBuffers[n][i].data()));
            }
        }
        else
        {
            DPU_FOREACH(this->set, dpu, n)
            {
                CHECK(dpu_prepare_xfer(dpu, arg.hostPtr));
            }
        }

        CHECK(dpu_push_xfer(this->set,
                            DPU_XFER_TO_DPU,
                            symbol.c_str(),
                            i * arg.elementSize,
                            arg.elementSize, // Already padded
                            DPU_XFER_DEFAULT));
    }

    return true;
}

bool Device::broadcastVectorToDpus(Arg &arg)
{
    if (!arg.isLookup())
        return false;

    dpu_set_t dpu;
    uint32_t n;
    uint32_t byteCount = Device::paddedSizeMram(arg.elementSize * arg.lookupLength);

    // Every DPU receives the whole lookup table at the arg's MRAM offset
    DEBUG_HOST("Lookup broadcast: %p[%lu] -> M+%lu [Size %d]",
               arg.hostPtr, arg.lookupLength, arg.mramOffset, byteCount);
    DPU_FOREACH(this->set, dpu, n)
    {
        CHECK(dpu_prepare_xfer(dpu, arg.hostPtr));
    }
    CHECK(dpu_push_xfer(this->set,
                        DPU_XFER_TO_DPU,
                        DPU_MRAM_HEAP_POINTER_NAME,
                        arg.mramOffset,
                        byteCount,
                        DPU_XFER_DEFAULT));

    return true;
}

bool Device::transferFromDpus(Arg &arg, uint64_t baseIndex, uint64_t elementsPerDpu)
{
    if (!arg.isVector())
        return false;

    dpu_set_t dpu;
    uint32_t n;

    DPU_FOREACH(this->set, dpu, n)
    {
        DEBUG_HOST("Output Transfer: M+%lu-> %p[%lu]  [DPU %d, Size %d, End Length Fraction: %d]",
                   arg.mramOffset,
                   arg.hostPtr,
                   arg.elementSize * (baseIndex + n * elementsPerDpu) / arg.endLengthFraction,
                   n,
                   Device::paddedSizeMram(arg.elementSize * elementsPerDpu / arg.endLengthFraction),
                   arg.endLengthFraction);
        CHECK(dpu_prepare_xfer(dpu, arg.hostPtr + arg.elementSize * (baseIndex + n * elementsPerDpu) / arg.endLengthFraction));
    }
    CHECK(dpu_push_xfer(this->set,
                        DPU_XFER_FROM_DPU,
                        DPU_MRAM_HEAP_POINTER_NAME,
                        arg.mramOffset,
                        Device::paddedSizeMram(arg.elementSize * elementsPerDpu / arg.endLengthFraction),
                        DPU_XFER_DEFAULT));

    return true;
}

bool Device::transferFromDpus(const std::string &symbol, std::vector<std::vector<uint32_t> > &dest)
{
    if (dest.size() != this->dpuCountUsed)
        return false;

    dpu_set_t dpu;
    uint32_t n;

    uint64_t length = dest[0].size();

    DPU_FOREACH(this->set, dpu, n)
    {
        if (dest[n].size() != length)
            return false;
        CHECK(dpu_prepare_xfer(dpu, dest[n].data()));
    }
    CHECK(dpu_push_xfer(this->set,
                        DPU_XFER_FROM_DPU,
                        symbol.c_str(),
                        0,
                        length * sizeof(uint32_t),
                        DPU_XFER_DEFAULT));

    return true;
}

bool Device::transferFromDpus(Arg &arg)
{
    if (!arg.isScalar())
        return false;

    dpu_set_t dpu;
    uint32_t n;
    std::string symbol = "argScalarRaw" + std::to_string(arg.index);

    // Only reduce accumulators (R banks each) are read back here
    for (uint32_t i = 0; i < arg.replicationFactor; i++)
    {
        DPU_FOREACH(this->set, dpu, n)
        {
            CHECK(dpu_prepare_xfer(dpu, arg.reduceBuffers[n][i].data()));
        }
        CHECK(dpu_push_xfer(this->set,
                            DPU_XFER_FROM_DPU,
                            symbol.c_str(),
                            i * arg.elementSize,
                            arg.elementSize, // Already padded
                            DPU_XFER_DEFAULT));
    }

    return true;
}

bool Device::transferScalarBytesFromDpus(const std::string &symbol, uint32_t elementSize,
                                          std::vector<std::vector<uint8_t> > &dest)
{
    if (dest.size() != this->dpuCountUsed)
        return false;

    dpu_set_t dpu;
    uint32_t n;
    DPU_FOREACH(this->set, dpu, n)
    {
        if (dest[n].size() != elementSize)
            dest[n].resize(elementSize);
        CHECK(dpu_prepare_xfer(dpu, dest[n].data()));
    }
    CHECK(dpu_push_xfer(this->set,
                        DPU_XFER_FROM_DPU,
                        symbol.c_str(),
                        0,
                        elementSize,
                        DPU_XFER_DEFAULT));
    return true;
}

void Device::detectDpus()
{
    if (Device::dpuCountAvailable == 0)
    {
        dpu_set_t set;
        DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, (std::string("backend=") + Device::backend).c_str(), &set));
        dpu_get_nr_dpus(set, &Device::dpuCountAvailable);
        dpu_free(set);
    }
}

}
