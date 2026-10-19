#ifndef DAPPA_ARG_H
#define DAPPA_ARG_H

#include <string>
#include "nlohmann/json.hpp"
#include "arg_flags.h"

namespace DaPPA
{

class Arg : public ArgFlagsBase
{
public:
    // Needed by std::map
    Arg();

    explicit Arg(ArgFlags flags);
    void init(uint32_t _index, uint32_t lengthFraction, uint32_t _dpuCountUsed);
    void initReduceBuffers(uint32_t _dpuCountUsed);

    nlohmann::json toJson();
    void compress(const std::vector<std::vector<uint32_t> > &filterOutputLengths, uint64_t baseIndex,
                  uint32_t elementsPerDpuThisRound, uint32_t taskletElementCount, uint32_t additionalElementTaskletCount, uint32_t dpuCountUsed);

    // Initialized by ArgTyped
    std::string typeStr;
    uint32_t elementSize;
    // Number of elements of a lookup table (0 for all other args)
    uint64_t lookupLength;
    uint8_t *hostPtr;
    std::function<void(void *, const void *)> combineFunc;
    std::vector<std::vector<std::vector<uint8_t> > > reduceBuffers;
    // Per-round, per-DPU scan chunk sums (raw bytes), used to stitch scan results on the host
    std::vector<std::vector<std::vector<uint8_t> > > scanChunkSums;

    // Initialized by init()
    uint32_t index;
    // Replication factor R of a scalar reduce accumulator, i.e., its number of WRAM banks.
    // R == taskletCountPerDpu gives every tasklet a private copy; a smaller R shares banks
    // between tasklets behind a mutex (used for large accumulators).
    uint32_t replicationFactor;
    uint64_t length;
    uint64_t cpuLength;
    uint32_t transferOverlap;
    uint32_t mramOverlap;
    uint32_t maxPadding;
    uint32_t startLengthFraction;
    uint32_t endLengthFraction;

    // Initialized by Pipeline
    uint64_t mramOffset;
    uint8_t *cpuPtr;

private:
    nlohmann::json json;
};

}

#endif //DAPPA_ARG_H
