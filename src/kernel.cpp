#include "dappa.h"
#include "kernel.h"
#include "device.h"
#include "logging.h"

namespace DaPPA
{

Kernel::Kernel(std::string str, std::function<void(const Kernel*)> _cpuFunc, uint32_t _groupSize, uint32_t _inputLengthFraction, KernelArg _filterOutput, std::list<KernelArg> _args)
        :
        kernelString(std::move(str)),
        cpuFunc(std::move(_cpuFunc)),
        args(std::move(_args)),
        filterOutput(std::move(_filterOutput)),
        stageOverlap(0),
        cacheElementCount(0),
        groupSize(_groupSize),
        inputLengthFraction(_inputLengthFraction),
        outputLengthFraction(_inputLengthFraction * _groupSize),
        reduce(false),
        scan(false)
{
    // Strip lambda elements from the kernel string
    this->stripKernelStringBack(this->stripKernelStringFront());
}

bool Kernel::isFilter() const
{
    return this->filterOutput.isValid();
}

bool Kernel::isReduce() const
{
    return this->reduce;
}

void Kernel::markReduce()
{
    this->reduce = true;
}

bool Kernel::isScan() const
{
    return this->scan;
}

void Kernel::markScan()
{
    this->scan = true;
}

// Strips the front part of a lambda string (brackets, surrounding parentheses)
uint32_t Kernel::stripKernelStringFront()
{
    uint32_t parenthesesCount = 0;
    for (char &c : this->kernelString)
    {
        switch (c)
        {
            case '(':
                parenthesesCount++;

                [[ fallthrough ]];
            case '[':
                c = ' ';
                break;

            case ']':
                c = ' ';
                return parenthesesCount;

            case ' ':
            case '\t':
            case '\n':
                break;

            default:
                return parenthesesCount;
        }
    }

    return parenthesesCount;
}

// Strips the back part of a lambda string (surrounding parentheses)
void Kernel::stripKernelStringBack(uint32_t parenthesesCount)
{
    for (auto c = this->kernelString.rbegin();
         c != this->kernelString.rend() && parenthesesCount > 0;
         c++)
    {
        if (*c == ')')
        {
            *c = ' ';
            parenthesesCount--;
        }
    }
}

void Kernel::calcCacheOffsets(uint32_t cacheSize)
{
    uint32_t inputElementSizeSum = 0;
    uint32_t outputElementSizeSum = 0;
    uint32_t smallestElementSize = std::numeric_limits<uint32_t>::max();
    std::list<KernelArg*> inArgs;
    std::list<KernelArg*> outArgs;
    std::list<KernelArg*> cacheableArgs;
    std::map<KernelArg*, KernelArg*> argCacheMap;

#ifdef DAPPA_SAFE_CACHE_ALLOCATION
    // Safe cache allocation that won't re-use input caches as output caches

    // Iterate over all arguments and mark every vector arguments as an argument that needs its own cache
    // Also sums up the type sizes for all vector arguments and records the smallest type that occurs
    for (auto &arg : this->args)
    {
        if (arg.isVector())
        {
            inputElementSizeSum += arg.base->elementSize;
            smallestElementSize = std::min(smallestElementSize, arg.base->elementSize);
            cacheableArgs.push_back(&arg);
        }
    }

    // Filter outputs are separate from the other arguments
    if (this->isFilter())
    {
        inputElementSizeSum += this->filterOutput.base->elementSize;
        smallestElementSize = std::min(smallestElementSize, this->filterOutput.base->elementSize);
        cacheableArgs.push_back(&this->filterOutput);
    }
#else
    // Iterate over all arguments and make a list of pure inputs and pure outputs. For now, only
    // inputs are marked as requiring a cache, outputs will be handled later.
    // Also sums up the type sizes for all inputs vector arguments and records the smallest type that occurs overall
    for (auto &arg : this->args)
    {
        if (!arg.isVector())
            continue;

        if (arg.isInput())
        {
            if (!arg.isOutput())
                inArgs.push_back(&arg);

            cacheableArgs.push_back(&arg);
            inputElementSizeSum += arg.base->elementSize;
        }
        else
        {
            outArgs.push_back(&arg);
        }
        smallestElementSize = std::min(smallestElementSize, arg.base->elementSize);
    }
    // Filter outputs are separate from the other arguments
    if (this->isFilter())
        cacheableArgs.push_back(&this->filterOutput);

    // Sort all input and output arguments
    inArgs.sort([](const auto &a, const auto &b){ return a->base->elementSize > b->base->elementSize; });
    outArgs.sort([](const auto &a, const auto &b){ return a->base->elementSize > b->base->elementSize; });

    // Iterate over all pure output args and try to find a pure input arg that it can share a cache with.
    // We sorted the inputs/outputs so that the largest output type will use the cache of the largest input
    // size in order not to waste any cache space.
    for (auto &outArg : outArgs)
    {
        bool found = false;
        for (auto inArg = inArgs.begin(); inArg != inArgs.end(); inArg++)
        {
            if ((*inArg)->base->elementSize * this->groupSize >= outArg->base->elementSize)
            {
                argCacheMap[outArg] = *inArg;
                inArgs.erase(inArg);
                found = true;
                break;
            }
        }
        // If there is no suitable input cache to put the current output in, we need to
        // allocate a new cache for it
        if (!found)
        {
            cacheableArgs.push_back(outArg);
            outputElementSizeSum += outArg->base->elementSize;
        }
    }
#endif

    // For the unlikely case that there are no vector arguments
    if (inputElementSizeSum == 0 && outputElementSizeSum == 0)
    {
        inputElementSizeSum = 1;
        smallestElementSize = 1;
    }

    // Calculate an approximation of how many elements of each input/output we can fit in the caches
    this->cacheElementCount = cacheSize / (inputElementSizeSum + outputElementSizeSum / this->groupSize);

    // Ensure that the calculated number of elements fit into cache after accounting for alignment restrictions
    this->cacheElementCount = this->alignedCount(this->cacheElementCount, cacheSize);

    // Clamp the cache element count to the maximum MRAM/WRAM transfer size, as any count higher than
    // that would require multiple transfers and has no performance benefits
    if (smallestElementSize * this->cacheElementCount > Device::maxWramTransferSize && Device::maxWramTransferSize % smallestElementSize == 0)
    {
        uint32_t preClamp = this->cacheElementCount;
        uint32_t clamped  = this->alignedCount(Device::maxWramTransferSize / smallestElementSize, cacheSize);
        // The clamp is only a DMA optimization. For groups with small outputs, alignment may
        // force the clamped count to 0, in which case we keep the (valid) unclamped count.
        this->cacheElementCount = (clamped > 0) ? clamped : preClamp;
    }

    // Windowed inputs read (cacheElementCount + inputOverlap) elements per block, so the
    // overlap also counts against the transfer size limit (avoids an extra DMA per block)
    for (auto &arg : cacheableArgs)
    {
        if (!arg->isInput() || arg->inputOverlap == 0)
            continue;
        uint32_t elemSize = arg->base->elementSize;
        if (elemSize == 0 || Device::maxWramTransferSize % elemSize != 0)
            continue;
        uint32_t maxCount = Device::maxWramTransferSize / elemSize;
        maxCount = maxCount > arg->inputOverlap ? maxCount - arg->inputOverlap : 0;
        if (this->cacheElementCount > maxCount)
            this->cacheElementCount = maxCount;
    }

    // Re-apply alignment after the overlap clamp so count * elementSize stays MRAM-aligned
    this->cacheElementCount = this->alignedCount(this->cacheElementCount, cacheSize);

    // A block size of 0 would make the DPU loop spin forever. This happens when a single
    // group does not fit into the per-tasklet WRAM cache.
    if (this->cacheElementCount == 0)
        throw std::runtime_error(
            "Kernel::calcCacheOffsets: cacheElementCount floored to 0 (groupSize=" +
            std::to_string(this->groupSize) + ", cacheSize=" + std::to_string(cacheSize) +
            "). One group does not fit the per-tasklet WRAM cache; reduce groupSize or "
            "increase available WRAM (e.g. shrink scalar parameters).");

    // Calculate the offset in cache memory from the base address for each argument
    uint32_t offset = 0;
    for (auto &arg : cacheableArgs)
    {
        arg->cacheOffset = offset;
        if (arg->isInput())
            offset += Device::paddedSizeMram(arg->base->elementSize * (this->cacheElementCount + arg->inputOverlap));
        else
            offset += arg->base->elementSize * this->cacheElementCount / this->groupSize;
        DEBUG_HOST("Arg %d cache offset: %d (size: %d  input overlap: %d)", arg->base->index, arg->cacheOffset, this->cacheElementCount, arg->inputOverlap);
    }
    // Set the offsets for the arguments that share a cache
    for (auto &[arg, target] : argCacheMap)
        arg->cacheOffset = target->cacheOffset;
}

// Produces a JSON object of the current kernel that can be used in the DPU code template
nlohmann::json Kernel::toJson()
{
    if (this->json.empty())
    {
        this->json["func"] = this->kernelString;
        this->json["isFilter"] = this->isFilter();
        this->json["isReduce"] = this->isReduce();
        this->json["isScan"] = this->isScan();
        this->json["cacheElementCount"] = this->cacheElementCount;
        this->json["stageOverlap"] = this->stageOverlap;
        this->json["groupSize"] = this->groupSize;
        this->json["inputLengthFraction"] = this->inputLengthFraction;
        this->json["outputLengthFraction"] = this->outputLengthFraction;
        this->json["filterInputIndex"] = -1;
        for (auto &arg : this->args)
        {
            if (arg.isVector() && arg.isInput() && arg.base->isFiltered())
            {
                this->json["filterInputIndex"] = arg.base->index;
                break;
            }
        }

        // Interleaved (strided) tasklet indexing is only valid for plain MAP and REDUCE
        // stages. Filter, scan, window and group stages need contiguous per-tasklet data.
        // Pipeline::buildJson further decides whether the whole pipeline can use it.
        this->json["interleaved"] = !this->isFilter()
                                 && !this->isScan()
                                 && this->groupSize == 1
                                 && this->stageOverlap == 0
                                 && this->inputLengthFraction == 1
                                 && this->json["filterInputIndex"].get<int>() < 0;
        // Scan stages: expose the INOUT output and the input vector directly to the template
        this->json["scanOutputIndex"] = -1;
        this->json["scanInputIndex"]  = -1;
        this->json["scanOutputType"]  = "";
        if (this->isScan())
        {
            for (auto &arg : this->args)
            {
                if (arg.isVector() && arg.isInput() && arg.isOutput())
                {
                    this->json["scanOutputIndex"] = arg.base->index;
                    this->json["scanOutputType"]  = arg.base->typeStr;
                    this->json["scanOutputArg"]   = arg.toJson();
                }
                else if (arg.isVector() && arg.isInput() && !arg.isOutput())
                {
                    this->json["scanInputIndex"] = arg.base->index;
                    this->json["scanInputArg"]   = arg.toJson();
                }
            }
        }

        // Add all arguments of the kernel for this stage
        uint32_t minArgSize = std::numeric_limits<uint32_t>::max();
        bool firstFilter = this->isFilter();
        for (auto &arg : this->args)
        {
            nlohmann::json argJson = arg.toJson();

            if (firstFilter)
            {
                firstFilter = false;

                // We have a different cache for that case
                if (arg.isInOut())
                    argJson["isOutput"] = false;
            }
            minArgSize = std::min(minArgSize, arg.base->elementSize);

            this->json["args"].push_back(argJson);
        }
        if (this->stageOverlap > 0)
            this->json["alignmentOverlap"] = static_cast<uint32_t>(std::ceil(static_cast<double>(Device::mramAlignment) / minArgSize));
        else
            this->json["alignmentOverlap"] = 0;

        nlohmann::json filterJson;
        if (this->isFilter())
        {
            filterJson = this->filterOutput.toJson();
        }
        else
        {
            filterJson["index"] = -1;
            filterJson["type"] = "nullptr_t";
            filterJson["elementSize"] = 0;
            filterJson["paddedElementSize"] = 0;
            filterJson["mramOffset"] = 0;
            filterJson["cacheOffset"] = 0;
            filterJson["isInput"] = false;
            filterJson["isOutput"] = false;
            filterJson["isScalar"] = false;
            filterJson["isVector"] = false;
        }
        this->json["filterOutput"] = filterJson;
        this->json["firstArg"] = this->json["args"][0];
    }

    return this->json;
}

// Reduces 'count' until all arguments fit into 'max' 'count' times while accounting for alignment
uint32_t Kernel::alignedCount(uint32_t count, uint32_t max)
{
    bool alignmentValid = false;
    while (!alignmentValid && count > 0)
    {
        alignmentValid = true;
        uint32_t sum = 0;
        for (auto &arg : this->args)
        {
            if (!arg.isVector())
                continue;

            // Make sure we always have a multiple of the group size
            if (count % this->groupSize != 0)
            {
                count--;
                alignmentValid = false;
                break;
            }

            // Ensure every element is aligned at the same time
            if (arg.isInput())
            {
                if ((arg.base->elementSize * count) % Device::mramAlignment != 0)
                {
                    count--;
                    alignmentValid = false;
                    break;
                }
                sum += arg.base->elementSize * count;
            }
            else
            {
                if ((arg.base->elementSize * count / this->groupSize) % Device::mramAlignment != 0)
                {
                    count--;
                    alignmentValid = false;
                    break;
                }
                sum += arg.base->elementSize * count / this->groupSize;
            }
        }
        // Ensure the size doesn't exceed the max
        if (sum > max)
        {
            count--;
            alignmentValid = false;
        }
    }

    return count;
}

}
