#include "arg.h"
#include "device.h"
#include "logging.h"

namespace DaPPA
{

Arg::Arg()
        :
        ArgFlagsBase(ArgFlags::Invalid),
        typeStr("nullptr_t"),
        hostPtr(nullptr),
        mramOffset(0),
        elementSize(0),
        lookupLength(0),
        cpuPtr(nullptr),
        index(0),
        replicationFactor(Device::taskletCountPerDpu),
        length(0),
        cpuLength(0),
        transferOverlap(0),
        mramOverlap(0),
        startLengthFraction(1),
        endLengthFraction(1),
        maxPadding(0)
{
}

Arg::Arg(ArgFlags flags)
        :
        ArgFlagsBase(flags),
        typeStr("nullptr_t"),
        hostPtr(nullptr),
        mramOffset(0),
        elementSize(0),
        lookupLength(0),
        cpuPtr(nullptr),
        index(0),
        replicationFactor(Device::taskletCountPerDpu),
        length(0),
        cpuLength(0),
        transferOverlap(0),
        mramOverlap(0),
        startLengthFraction(1),
        endLengthFraction(1),
        maxPadding(0)
{
}

void Arg::init(uint32_t _index, uint32_t lengthFraction, uint32_t _dpuCountUsed)
{
    this->index = _index;
    this->startLengthFraction = lengthFraction;
    this->endLengthFraction = lengthFraction;

    // reduceBuffers are allocated in initReduceBuffers() once R is known
    if (this->isScalar() && this->isOutput() && _dpuCountUsed > 0)
        this->initReduceBuffers(_dpuCountUsed);
}

// Allocates R accumulator banks per DPU, each initialized with the host-side value.
// Called again by Pipeline::calcElementCounts if R changes.
void Arg::initReduceBuffers(uint32_t _dpuCountUsed)
{
    if (!(this->isScalar() && this->isOutput()) || _dpuCountUsed == 0)
        return;

    std::vector<uint8_t> initVector(this->elementSize);
    std::memcpy(initVector.data(), this->hostPtr, this->elementSize);

    this->reduceBuffers.assign(_dpuCountUsed, {});
    for (auto &bankBuffers : this->reduceBuffers)
    {
        bankBuffers.assign(this->replicationFactor, initVector);
    }
}

// Produces a JSON object of the current arg that can be used in the DPU code template
nlohmann::json Arg::toJson()
{
    if (this->json.empty())
    {
        this->json["index"] = this->index;
        this->json["type"] = this->typeStr;
        this->json["elementSize"] = this->elementSize;
        this->json["paddedElementSize"] = Device::paddedSizeMram(this->elementSize);
        this->json["mramOffset"] = this->mramOffset;
        this->json["inputOverlap"] = this->transferOverlap;
        this->json["startLengthFraction"] = this->startLengthFraction;
        this->json["endLengthFraction"] = this->endLengthFraction;
        this->json["isInput"] = this->isInput();
        this->json["isOutput"] = this->isOutput();
        this->json["isScalar"] = this->isScalar();
        this->json["isVector"] = this->isVector();
        this->json["isFiltered"] = this->isFiltered();
        this->json["isWindowed"] = this->isWindowed();
        this->json["isScanned"]  = this->isScanned();
        this->json["isReduced"]  = this->isReduced();
        this->json["isLookup"]   = this->isLookup();
        // Number of WRAM accumulator banks (R) of a reduce output
        this->json["replicationFactor"] = this->replicationFactor;
        // Banks shared between tasklets need a mutex
        this->json["isReduceShared"] = this->isScalar() && this->isOutput()
                                       && (this->replicationFactor < Device::taskletCountPerDpu);
    }

    return this->json;
}

// Compress the results of a filter operation to remove any 'holes' in the data
void Arg::compress(const std::vector<std::vector<uint32_t> > &filterOutputLengths, uint64_t baseIndex,
                   uint32_t elementsPerDpuThisRound, uint32_t taskletElementCount, uint32_t additionalElementTaskletCount, uint32_t dpuCountUsed)
{
    // Iterate over each tasklet in each DPU, as they all have their own buffers
    for (uint32_t d = 0; d < dpuCountUsed; d++)
    {
        for (uint32_t t = 0; t < Device::taskletCountPerDpu; t++)
        {
            // Calculate the offset of the current tasklet within the current DPU
            uint32_t taskletOffset;
            if (t < additionalElementTaskletCount)
                taskletOffset = t * taskletElementCount;
            else
                taskletOffset = additionalElementTaskletCount * (taskletElementCount + 1) + (t - additionalElementTaskletCount) * taskletElementCount;

            // Calculate the pointer to the start of the data we received from the current tasklet
            void *srcPtr = this->hostPtr + this->elementSize * (baseIndex
                                                                + d * elementsPerDpuThisRound
                                                                + taskletOffset)
                                                               / this->endLengthFraction;

            // Calculate the pointer to the end of contiguous data block that we will move the result to
            void *dstPtr = this->hostPtr + this->length * this->elementSize;

            // Retrieve the data size for the current tasklet
            auto lengthTasklet = filterOutputLengths[d][t];

            // Move the data
            if (srcPtr != dstPtr)
            {
                DEBUG_HOST("Compressing arg %d: %d -> %ld (length %d)",
                           this->index,
                           (baseIndex
                            + d * elementsPerDpuThisRound
                            + taskletOffset)
                           / this->endLengthFraction,
                           this->length,
                           lengthTasklet);
                std::memmove(dstPtr, srcPtr, lengthTasklet * this->elementSize);
            }
            this->length += lengthTasklet;
        }
    }
}

}
