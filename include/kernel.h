#ifndef DAPPA_KERNEL_H
#define DAPPA_KERNEL_H

#include <string>
#include <vector>
#include <list>
#include <functional>
#include "kernel_arg.h"

namespace DaPPA {

class Kernel
{
public:
    Kernel() = delete;
    Kernel(std::string str, std::function<void(const Kernel*)> _cpuFunc, uint32_t _groupSize, uint32_t _inputLengthFraction, KernelArg _filterOutput, std::list<KernelArg> _args);

    [[ nodiscard ]] bool isFilter() const;
    [[ nodiscard ]] bool isReduce() const;
    [[ nodiscard ]] bool isScan() const;
    void markReduce();
    void markScan();
    [[nodiscard]] nlohmann::json toJson();
    void calcCacheOffsets(uint32_t cacheSize);

    std::string kernelString;
    std::list<KernelArg> args;
    KernelArg filterOutput;
    std::function<void(const Kernel*)> cpuFunc;
    uint32_t cacheElementCount;
    uint32_t stageOverlap;
    uint32_t groupSize;
    uint32_t inputLengthFraction;
    uint32_t outputLengthFraction;
    nlohmann::json json;

private:
    bool reduce;
    bool scan;

    uint32_t stripKernelStringFront();
    void stripKernelStringBack(uint32_t parenthesesCount);
    uint32_t alignedCount(uint32_t count, uint32_t max);
};

}

#endif //DAPPA_KERNEL_H
