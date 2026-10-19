#include "pipeline.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>
#include <fstream>
#include <random>
#include <sstream>
#include <filesystem>
#include <string_view>
#include "inja.hpp"
#include "device.h"
#include "shared.h"
#include "dappa.h"

// Read files into variables at compile time
extern "C" {
    #define INCBIN_PREFIX
    #include "incbin.h"
    INCTXT(deviceTemplateBin, DAPPA_SOURCE_DIR "/src/device_template.c.inja");
    INCTXT(loggingHeaderBin, DAPPA_SOURCE_DIR "/include/logging.h");
    INCTXT(sharedHeaderBin, DAPPA_SOURCE_DIR "/include/shared.h");
}

#define CHECK(x) do { if (!(x)){ this->hasFailed = true; cpuThread.join(); device.readLog(); return false; } } while(0)

namespace DaPPA
{

Pipeline::Pipeline(uint64_t _dataLength)
        :
        dataLength(_dataLength),
        hasFailed(false),
        totalElementsPerDpu(0),
        maxElementsPerDpuPerRound(0),
        totalElementsPerTasklet(0),
        taskletMaxElementsPerRound(0),
        cpuElementCount(0),
        maxGroupSize(1),
        cacheSize(0),
        segmentLength(1)
{
    this->timingPoint();
    this->headers.emplace_back(loggingHeaderBinData);
    this->headers.emplace_back(sharedHeaderBinData);

    Device::detectDpus();

    // In case of very short args, we might not need every DPU
    this->dpuCountUsed = std::min<uint64_t>(this->dataLength / Device::taskletCountPerDpu, Device::dpuCountAvailable);

    // DAPPA_EMIT_RECIPE=<dir>: write the DPU program and its inputs for offline simulation instead of running
    if (const char *env = std::getenv("DAPPA_EMIT_RECIPE"))
        this->recipeDir = env;

    // DAPPA_EMIT_BBOP=<dir>: emit a bbop harness for the SIMDRAM cost model instead of running
    if (const char *env = std::getenv("DAPPA_EMIT_BBOP"))
        this->bbopDir = env;

    this->timingPoint("Constructor");
}

Pipeline::~Pipeline()
{
#ifdef MEASURE_PERFORMANCE_HOST
    auto last = this->timingPoints.begin();
    if (last  == this->timingPoints.end())
        return;
    auto next = std::next(last);
    std::chrono::duration<double, std::milli> total = this->timingPoints.rbegin()->timePoint - this->timingPoints.begin()->timePoint;

    std::cout.precision(2);
    std::cout << std::fixed;
#ifndef MEASURE_PERFORMANCE_NO_HEADERS
    std::cout << "Timing info" << std::endl;
    std::cout << "-----------" << std::endl;
#endif
    while (next != this->timingPoints.end())
    {
        if (!next->name.empty())
        {
            auto reference = last->timePoint;
            if (!next->reference.empty())
            {
                auto search = std::make_reverse_iterator(next);
                while (search != this->timingPoints.rend())
                {
                    if (search->name == next->reference)
                    {
                        reference = search->timePoint;
                        break;
                    }
                    search++;
                }
                if (search == this->timingPoints.rend())
                {
                    std::cerr << "Error: Invalid timing point reference";
                    return;
                }
            }

            std::chrono::duration<double, std::milli> duration = next->timePoint - reference;
            double percentage = 100.0 * (duration.count() / total.count());
            std::cout << next->name << ": " << duration.count() << "ms (" << percentage << "%)" << std::endl;
        }

        next++;
        last++;
    }
#ifndef MEASURE_PERFORMANCE_NO_HEADERS
    std::cout << "--------------------" << std::endl;
#endif
    std::cout << "Total time: " << total.count() << "ms" << std::endl;
#endif
}

void Pipeline::addHeader(const std::string &data)
{
    this->headers.push_back(data);
}

uint64_t Pipeline::getLength(void *ptr)
{
    if (this->hasFailed)
        return 0;

    auto arg = this->args.find(static_cast<uint8_t*>(ptr));
    if (arg == this->args.end())
    {
        ERROR_HOST("Argument not found");
        return 0;
    }

    if (arg->second.isFiltered())
        return arg->second.length;
    else
        return this->dataLength / arg->second.endLengthFraction;
}

bool Pipeline::execute()
{
    this->timingPoint("Execute start");

    // Very basic sanity check
    if (this->dataLength == 0)
    {
        this->hasFailed = true;
        ERROR_HOST("Input length is zero");
        return false;
    }
    if (this->hasFailed)
    {
        ERROR_HOST("Pipeline failed previously");
        return false;
    }

    // The processing-using-DRAM backend does its own tiling, so skip the UPMEM-specific layout
    if (!this->bbopDir.empty())
    {
        this->emitBbop();
        return true;
    }

    // Calculate the number of elements per Tasklet/Device/CPU
    this->calcElementCounts();

    // Calculate the mramOffset of all args elements
    // Note: Must be called *after* Pipeline::calcElementCounts()
    this->calcOffsets();

    // Calculate cache offsets for each kernel
    for (auto &kernel : this->stages)
        kernel.calcCacheOffsets(this->cacheSize);

    this->timingPoint("Pipeline calculations");

    // Build the JSON object for the template
    nlohmann::json json = this->buildJson();

    // Create a temporary directory for the compilation
    auto tempDir = createTempDir();

    // Generate the device code
    inja::Environment injaEnv;
#if VERBOSITY >= VERBOSITY_DEBUG_ALL
    injaEnv.add_callback("struct_type", 1, [](const inja::Arguments &iargs) {
        return iargs.at(0)->get<std::string_view>().at(6) == ' ';
    });
#endif
    injaEnv.set_trim_blocks(true);
    injaEnv.set_lstrip_blocks(true);
    inja::Template deviceTemplate = injaEnv.parse(deviceTemplateBinData);
    injaEnv.write(deviceTemplate, json, tempDir + "/device.c");

    this->timingPoint("Templating");

    // Recipe mode: dump the device code and inputs (e.g., for uPIMulator) without compiling or launching
    if (!this->recipeDir.empty())
    {
        std::filesystem::create_directories(this->recipeDir);
        std::filesystem::copy_file(tempDir + "/device.c",
                                   this->recipeDir + "/device.c",
                                   std::filesystem::copy_options::overwrite_existing);
        this->emitRecipe();
        this->timingPoint("Recipe emit");
        return true;
    }

    // Compile the generated device code
    char compileCommand[1024];
#if VERBOSITY >= VERBOSITY_DEBUG
    const auto optimizationLevel = "-O0";
#else
    const auto optimizationLevel = "-O3";
#endif
    std::snprintf(compileCommand, 1024, "dpu-upmem-dpurte-clang -working-directory=%s %s -DNR_TASKLETS=%d -DSTACK_SIZE_DEFAULT=%d -o device device.c", tempDir.c_str(), optimizationLevel, Device::taskletCountPerDpu, Device::taskletStackSize);
    if (std::system(compileCommand) != 0)
    {
        this->hasFailed = true;
        ERROR_HOST("Failed to compile DPU binary");
        return false;
    }

    this->timingPoint("Compilation");

    // Start working on the CPU part
    std::thread cpuThread(&Pipeline::cpuWorker, this);
    this->timingPoint("CPU thread start");

    // In case the data isn't large enough to run on DPUs
    if (this->dpuCountUsed > 0)
    {
        // Prepare the DPUs
        Device device(tempDir + "/device", this->dpuCountUsed);
        this->timingPoint("Device preparations");

#if VERBOSITY < VERBOSITY_DEBUG
        // Remove the temp data
        std::filesystem::remove_all(tempDir);
        this->timingPoint("Temp data removal");
#endif

        // Lookup args are round-invariant, so broadcast them once before the first round
        for (auto &[ptr, arg] : this->args)
        {
            if (arg.isLookup() && arg.isInput() && arg.isPushed())
                CHECK(device.broadcastVectorToDpus(arg));
        }

        // Iterate through the rounds until all args were handled
        uint64_t elementsLeftPerDpu = this->totalElementsPerDpu;
        for (uint32_t round = 0; elementsLeftPerDpu > 0; round++)
        {
            DEBUG_HOST("Round %d start\n-------------", round);

            // Calculate how many taskletElementsThisRound the tasklets will process this round
            uint32_t elementsPerDpuThisRound = std::min<uint64_t>(elementsLeftPerDpu, this->maxElementsPerDpuPerRound);
            elementsLeftPerDpu -= elementsPerDpuThisRound;
            uint64_t roundBaseArgIndex = static_cast<uint64_t>(round) * this->maxElementsPerDpuPerRound * this->dpuCountUsed;

            HostData hostData;
            hostData.taskletElementCount = elementsPerDpuThisRound / Device::taskletCountPerDpu;
            hostData.additionalElementTaskletCount = elementsPerDpuThisRound % Device::taskletCountPerDpu;
            CHECK(device.broadcastToDpus("hostData", hostData));
            DEBUG_HOST("Tasklet element count: %d  Additional element tasklet count: %d", hostData.taskletElementCount, hostData.additionalElementTaskletCount);

            DEBUG_HOST("Elements per DPU: %d", elementsPerDpuThisRound);
            this->timingPoint("Round preparations");

            // Copy the inputs to the DPUs
            for (auto &[ptr, arg] : this->args)
            {
                if (arg.isInput() && arg.isPushed())
                {
                    if (arg.isLookup())
                    {
                        // Already broadcast before the first round
                        continue;
                    }
                    else if (arg.isVector())
                    {
                        CHECK(device.transferToDpus(arg, roundBaseArgIndex, elementsPerDpuThisRound));
                    }
                    else
                    {
                        CHECK(device.broadcastToDpus(arg));
                    }
                }
            }
            this->timingPoint("Copy inputs");

            // Run the DPUs
            CHECK(device.launch());
            this->timingPoint("DPU run");

#if VERBOSITY >= VERBOSITY_DEBUG || defined MEASURE_PERFORMANCE_DPU
            device.readLog();
#endif

            // Copy the args back from the DPUs
            for (auto &[ptr, arg] : this->args)
            {
                if (arg.isFetched() && arg.isOutput())
                {
                    if (arg.isVector())
                    {
                        std::vector<std::vector<uint32_t> > filterOutputLengths;
                        if (arg.isFiltered())
                        {
                            // Get the correct length to transfer back
                            filterOutputLengths = {this->dpuCountUsed, std::vector<uint32_t>(Device::taskletCountPerDpu)};
                            std::string symbol = "filterOutputLengthArg" + std::to_string(arg.index);
                            device.transferFromDpus(symbol, filterOutputLengths);
                        }

                        CHECK(device.transferFromDpus(arg, roundBaseArgIndex, elementsPerDpuThisRound));

                        if (arg.isFiltered() && arg.isFiltered())
                            arg.compress(filterOutputLengths, roundBaseArgIndex, elementsPerDpuThisRound, hostData.taskletElementCount, hostData.additionalElementTaskletCount, this->dpuCountUsed);
                    }
                    else
                    {
                        CHECK(device.transferFromDpus(arg));
                    }
                }

                // Collect the per-DPU chunk sums of scanned args to stitch the scan together later
                if (arg.isScanned())
                {
                    arg.scanChunkSums.resize(round + 1);
                    arg.scanChunkSums[round].assign(this->dpuCountUsed, std::vector<uint8_t>(arg.elementSize));
                    std::string symbol = "scanChunkSumArg" + std::to_string(arg.index);
                    CHECK(device.transferScalarBytesFromDpus(symbol, arg.elementSize, arg.scanChunkSums[round]));
                }
            }
            this->timingPoint("Copy outputs");
        }
    }

    cpuThread.join();
    this->timingPoint("Device calculations", "Device preparations");

    this->compressFilteredCpuArgs();
    this->timingPoint("Compression");

    this->combineReductions();
    this->timingPoint("Combination");

    this->combineScans();
    this->timingPoint("Scan stitch");

    this->timingPoint("Execution", "Execute start");

    return true;
}

void Pipeline::fetch(void *ptr, bool silent)
{
    auto arg = this->args.find(static_cast<uint8_t*>(ptr));
    if (arg != this->args.end())
    {
        arg->second.markFetched();
    }
    else if (!silent)
    {
        ERROR_HOST("Argument not found");
    }
}

// Note: Must be called *after* Pipeline::calcElementCounts()
void Pipeline::calcOffsets()
{
    // Each vector arg's MRAM region only needs to hold the largest single round. If
    // all data fits in one round, reserving a full round would waste MRAM and can
    // push the offsets of later args past the end of MRAM.
    uint64_t elementsPerRound = std::min<uint64_t>(this->maxElementsPerDpuPerRound, this->totalElementsPerDpu);
    uint64_t offset = 0;
    for (auto &[ptr, arg] : this->args)
    {
        if (arg.isVector())
        {
            arg.mramOffset = offset;
            DEBUG_HOST("Arg %d MRAM offset: %ld", arg.index, arg.mramOffset);
            offset += Device::paddedSizeMram(arg.elementSize * (elementsPerRound / arg.startLengthFraction + arg.mramOverlap));
        }
        else if (arg.isLookup())
        {
            // The whole lookup array stays in MRAM for all rounds
            arg.mramOffset = offset;
            DEBUG_HOST("Arg %d lookup MRAM offset: %ld (length %ld)", arg.index, arg.mramOffset, arg.lookupLength);
            offset += Device::paddedSizeMram(arg.elementSize * arg.lookupLength);
        }
    }
}

void Pipeline::calcElementCounts()
{
    // Determine the WRAM replication factor R of every reduce accumulator:
    //  - R = taskletCountPerDpu: one private copy per tasklet, no locking
    //  - 1 <= R < taskletCountPerDpu: R shared copies, each protected by a mutex
    //  - Does not fit even once: R = 1, and the WRAM check below disables the DPUs
    // Only large accumulators (e.g., big histograms) are shared, so that enough
    // WRAM is left for the input cache.
    {
        const uint32_t fixedPerTasklet = Device::taskletStackSize + Device::wramBuffer;
        // Keep at least one DMA block of input cache per tasklet
        const uint32_t minCacheTotal = Device::maxWramTransferSize * Device::taskletCountPerDpu;
        uint32_t inputScalarBytes = 0;
        for (auto &[ptr, arg] : this->args)
            if (!arg.isVector() && !arg.isLookup() && !(arg.isScalar() && arg.isOutput()))
                inputScalarBytes += Device::paddedSizeWram(Device::taskletCountPerDpu * arg.elementSize);

        const int64_t accumBudget = static_cast<int64_t>(Device::wramSize)
                                  - static_cast<int64_t>(fixedPerTasklet) * Device::taskletCountPerDpu
                                  - static_cast<int64_t>(minCacheTotal)
                                  - static_cast<int64_t>(inputScalarBytes);

        for (auto &[ptr, arg] : this->args)
        {
            if (!(arg.isScalar() && arg.isOutput()))
                continue;
            uint32_t maxBanks = accumBudget > 0
                ? static_cast<uint32_t>(accumBudget / Device::paddedSizeWram(arg.elementSize))
                : 0;
            uint32_t R = std::min<uint32_t>(Device::taskletCountPerDpu, maxBanks);
            arg.replicationFactor = (R == 0) ? 1 : R;
            DEBUG_HOST("Arg %d reduce replication factor R=%d (elementSize=%d, budget=%ld)",
                       arg.index, arg.replicationFactor, arg.elementSize, (long)accumBudget);
        }
    }

    // Calculate the smallest unit that has all outputs aligned (segment)
    bool alignmentValid = false;
    uint32_t segmentSize;
    uint32_t scalarSizeSum;
    uint32_t maxOverlap = 0;
    this->segmentLength = 1;
    while (!alignmentValid)
    {
        alignmentValid = true;
        segmentSize = 0;
        scalarSizeSum = 0;
        for (auto &[ptr, arg] : this->args)
        {
            // Lookup args live in MRAM and are not streamed
            if (arg.isLookup())
                continue;

            if (!arg.isVector())
            {
                // Reduce accumulators are stored R times, read-only scalars once
                uint32_t banks = (arg.isScalar() && arg.isOutput())
                    ? arg.replicationFactor : 1;
                scalarSizeSum += Device::paddedSizeWram(banks * arg.elementSize);
                continue;
            }

            if (this->segmentLength % arg.endLengthFraction != 0
                || (arg.elementSize * (this->segmentLength / arg.endLengthFraction)) % Device::mramAlignment != 0)
            {
                alignmentValid = false;
                this->segmentLength++;
                break;
            }
            segmentSize += arg.elementSize * this->segmentLength / arg.endLengthFraction;
            maxOverlap = std::max(maxOverlap, arg.mramOverlap);
        }
    }

    // For the unlikely case that there are no vector arguments
    if (segmentSize == 0)
        segmentSize = 1;

    // Calculate an approximation for the max number of elements per round
    this->taskletMaxElementsPerRound = this->segmentLength * (Device::mramSize / (segmentSize * Device::taskletCountPerDpu));

    // Calculate the cache size per tasklet
    this->cacheSize = ((Device::wramSize - scalarSizeSum) / Device::taskletCountPerDpu - Device::taskletStackSize - Device::wramBuffer) & ~(Device::mramAlignment - 1);

    // Lookup arrays reduce the MRAM space available for streamed vectors
    uint64_t lookupMramBytes = 0;
    for (auto &[ptr, arg] : this->args)
        if (arg.isLookup())
            lookupMramBytes += Device::paddedSizeMram(arg.elementSize * arg.lookupLength);

    // Ensure that the data fits into MRAM including the overlap
    uint64_t sum;
    alignmentValid = false;
    while (!alignmentValid && this->taskletMaxElementsPerRound > 0)
    {
        sum = lookupMramBytes;
        alignmentValid = true;
        for (auto &[ptr, arg] : this->args)
            if (arg.isVector())
                sum += Device::paddedSizeMram(Device::taskletCountPerDpu * arg.elementSize * (this->taskletMaxElementsPerRound / arg.endLengthFraction) + arg.elementSize * arg.mramOverlap);
        if (sum > Device::mramSize)
        {
            this->taskletMaxElementsPerRound -= this->segmentLength;
            alignmentValid = false;
        }
    }

    // Ensure that we have enough data to perform the windowing and grouping
    if (this->taskletMaxElementsPerRound == 0 || scalarSizeSum > Device::wramSize)
        this->dpuCountUsed = 0;
    else
        this->dpuCountUsed = std::min<uint64_t>((this->dataLength - maxOverlap) / this->segmentLength, Device::dpuCountAvailable);

    // Calculate the amount of elements per Device/tasklet
    if (this->dpuCountUsed == 0)
        this->totalElementsPerTasklet = 0;
    else
        this->totalElementsPerTasklet = static_cast<uint64_t>(this->dataLength * (1 - DAPPA_TARGET_CPU_LOAD)) / (this->dpuCountUsed * Device::taskletCountPerDpu);

    // Make sure the end of the data is aligned, or the last transferToDpu will overwrite CPU data
    this->totalElementsPerTasklet = (this->totalElementsPerTasklet / this->segmentLength) * this->segmentLength;

    this->totalElementsPerDpu = this->totalElementsPerTasklet * Device::taskletCountPerDpu;
    this->maxElementsPerDpuPerRound = this->taskletMaxElementsPerRound * Device::taskletCountPerDpu;
    this->cpuElementCount = this->dataLength - (this->totalElementsPerDpu * this->dpuCountUsed);

    // Resize the reduce buffers now that R and dpuCountUsed are final
    if (this->dpuCountUsed > 0)
        for (auto &[ptr, arg] : this->args)
            if (arg.isScalar() && arg.isOutput())
                arg.initReduceBuffers(this->dpuCountUsed);

    DEBUG_HOST("DPU count used: %d", this->dpuCountUsed);
    DEBUG_HOST("DPU element count: %lu  Tasklet element count: %lu", this->totalElementsPerDpu, this->totalElementsPerTasklet);
    DEBUG_HOST("DPU element count per round: %d  Tasklet max element count per round: %d", this->maxElementsPerDpuPerRound, this->taskletMaxElementsPerRound);
    DEBUG_HOST("CPU element count: %lu", this->cpuElementCount);
    DEBUG_HOST("Tasklet cache size: %d", this->cacheSize);
}

nlohmann::json Pipeline::buildJson()
{
    nlohmann::json json;

    json["cacheSize"] = this->cacheSize;
    json["taskletMaxElementCount"] = this->taskletMaxElementsPerRound;
    json["headers"] = this->headers;

    // Add the global list of args to reduce duplication in the resulting code
    uint32_t i = 0;
    for (auto &[ptr, arg] : this->args)
    {
        json["args"].push_back(arg.toJson());
        i++;
    }

    for (auto &kernel : this->stages)
        json["stages"].push_back(kernel.toJson());

    // Interleaved (PrIM-style strided) indexing requires all stages to use the same
    // block size, so that each tasklet owns the same elements in every stage. It is
    // only enabled if every stage supports it (plain map or reduce stages). The block
    // size is the smallest cache element count, rounded down to keep all MRAM
    // accesses aligned.
    bool allInterleavable = !this->stages.empty();
    for (auto &stage : json["stages"])
        if (!stage.value("interleaved", false))
            allInterleavable = false;

    uint32_t interleaveBlock = 0;
    if (allInterleavable)
    {
        uint32_t minCache = std::numeric_limits<uint32_t>::max();
        uint32_t alignMultiple = 1;
        for (auto &kernel : this->stages)
        {
            minCache = std::min(minCache, kernel.cacheElementCount);
            for (auto &arg : kernel.args)
            {
                if (!arg.isVector())
                    continue;
                uint32_t e = arg.base->elementSize;
                uint32_t need = static_cast<uint32_t>(Device::mramAlignment)
                              / std::gcd(static_cast<uint32_t>(Device::mramAlignment), e);
                alignMultiple = std::lcm(alignMultiple, need);
            }
        }
        interleaveBlock = (minCache / alignMultiple) * alignMultiple;
        if (interleaveBlock == 0)
            allInterleavable = false;
    }

    if (!allInterleavable)
        for (auto &stage : json["stages"])
            stage["interleaved"] = false;

    json["interleaveBlock"] = interleaveBlock;

    DEBUG_HOST_VV("JSON: %s", json.dump(2).c_str());

    return json;
}

void Pipeline::cpuWorker()
{
    std::list<std::unique_ptr<uint8_t[]> > ptrList;

    // Allocate buffers for non-fetched args
    for (auto &[ptr, arg] : this->args)
    {
        if (arg.isVector())
        {
            // Calculate a pointer that points to the start of the data that will be processed by the CPU
            uint8_t *cpuPtr = arg.hostPtr + this->totalElementsPerDpu * this->dpuCountUsed * arg.elementSize / arg.startLengthFraction;
            arg.cpuLength = this->cpuElementCount / arg.endLengthFraction;
            DEBUG_HOST_V("CPU arg %d ptr offset: %ld (%ld length, %d padding)", arg.index, this->totalElementsPerDpu * this->dpuCountUsed * arg.elementSize / arg.startLengthFraction, arg.cpuLength, arg.maxPadding);

            // Decide whether we need to copy the data in a separate buffer or not
            if ((arg.isFetched() || !arg.isOutput()) && arg.maxPadding == 0)
            {
                arg.cpuPtr = cpuPtr;
            }
            else
            {
                arg.cpuPtr = new uint8_t[this->cpuElementCount * arg.elementSize / arg.startLengthFraction + arg.maxPadding * arg.elementSize];
                ptrList.emplace_back(arg.cpuPtr);
                if (arg.isInput() && arg.isPushed())
                    std::memcpy(arg.cpuPtr, cpuPtr, this->cpuElementCount * arg.elementSize / arg.startLengthFraction);
            }
        }
        else if (arg.isScalar())
        {
            // Decide whether we need to copy the data in a separate buffer or not
            if (arg.isFetched() || !arg.isOutput())
            {
                arg.cpuPtr = arg.hostPtr;
            }
            else
            {
                arg.cpuPtr = new uint8_t[arg.elementSize];
                ptrList.emplace_back(arg.cpuPtr);
                if (arg.isInput() && arg.isPushed())
                    std::memcpy(arg.cpuPtr, arg.hostPtr, arg.elementSize);
            }
        }
        else if (arg.isLookup())
        {
            // Read-only, so the CPU can use the host array directly
            arg.cpuPtr = arg.hostPtr;
        }
    }

    // Run every kernel consecutively
    for (const auto &kernel : this->stages)
        kernel.cpuFunc(&kernel);
}

// https://stackoverflow.com/a/58454949
// Creates a temporary directory to store the DPU binary in
std::string Pipeline::createTempDir(uint32_t maxTries)
{
#if VERBOSITY >= VERBOSITY_DEBUG
    return ".";
#else
    auto tmpDir = std::filesystem::temp_directory_path();
    std::random_device dev;
    std::mt19937 prng(dev());
    std::uniform_int_distribution<uint64_t> rand(0);
    std::filesystem::path path;
    uint32_t i = 0;
    while (true) {
        std::stringstream ss;
        ss << "dappa_" << std::hex << rand(prng);
        path = tmpDir / ss.str();
        if (std::filesystem::create_directory(path)) {
            break;
        }
        if (i == maxTries) {
            return "";
        }
        i++;
    }
    return path;
#endif
}

// Moves the result of the CPU calculations to the end of the valid data chunk
void Pipeline::compressFilteredCpuArgs()
{
    for (auto &[ptr, arg] : this->args)
    {
        if (arg.isVector() && arg.isFiltered() && arg.isFetched())
        {
            void *dstPtr = ptr + arg.length * arg.elementSize;
            void *cpuPtr = ptr + this->totalElementsPerDpu * this->dpuCountUsed * arg.elementSize / arg.startLengthFraction;
            if (cpuPtr != dstPtr)
            {
                DEBUG_HOST_V("Compressing CPU base %d: %p -> %p (length %ld)", arg.index, cpuPtr, dstPtr, arg.cpuLength);
                std::memmove(dstPtr, cpuPtr, arg.cpuLength * arg.elementSize);
            }
            arg.length += arg.cpuLength;
        }
    }
}

// Combines the intermediate reduction results of all tasklets into a single value
void Pipeline::combineReductions()
{
    for (auto &[ptr, arg] : this->args)
        if (arg.isScalar() && arg.isOutput() && arg.isFetched())
            for (const auto &bankBuffers : arg.reduceBuffers)   // Per DPU
                for (const auto &buffer : bankBuffers)           // Per copy (R per DPU)
                    arg.combineFunc(arg.hostPtr, buffer.data());
}

// Turns the per-(round, DPU) local scans into a global scan. Each block is offset by
// the running fold of the chunk sums of all preceding blocks. The elements processed
// by the CPU are then scanned serially, starting from the last DPU result.
void Pipeline::combineScans()
{
    for (auto &kernel : this->stages)
    {
        if (!kernel.isScan())
            continue;

        // Find the scan output (InOut) and input vectors
        Arg *scanOut = nullptr;
        Arg *scanIn  = nullptr;
        for (auto &karg : kernel.args)
        {
            if (karg.isVector() && karg.isInput() && karg.isOutput())
                scanOut = karg.base;
            else if (karg.isVector() && karg.isInput() && !karg.isOutput())
                scanIn = karg.base;
        }
        if (!scanOut || !scanOut->combineFunc || scanOut->scanChunkSums.empty())
            continue;

        // Apply the offsets to the DPU results, in host order
        std::vector<uint8_t> cumulative(scanOut->elementSize, 0);
        const uint64_t totalElems = this->dataLength;
        const uint32_t maxPerDpu  = this->maxElementsPerDpuPerRound;
        uint64_t elementsLeftPerDpu = this->totalElementsPerDpu;
        uint64_t hostPos = 0;

        for (uint32_t round = 0;
             round < scanOut->scanChunkSums.size() && elementsLeftPerDpu > 0;
             round++)
        {
            uint32_t elementsThisRound = std::min<uint64_t>(elementsLeftPerDpu, maxPerDpu);
            elementsLeftPerDpu -= elementsThisRound;

            for (uint32_t d = 0; d < this->dpuCountUsed; d++)
            {
                for (uint32_t c = 0; c < elementsThisRound && hostPos < totalElems; c++)
                {
                    uint8_t *elem = scanOut->hostPtr + hostPos * scanOut->elementSize;
                    std::vector<uint8_t> tmp(scanOut->elementSize);
                    std::memcpy(tmp.data(), cumulative.data(), scanOut->elementSize);
                    scanOut->combineFunc(tmp.data(), elem);
                    std::memcpy(elem, tmp.data(), scanOut->elementSize);
                    hostPos++;
                }
                if (d < scanOut->scanChunkSums[round].size())
                    scanOut->combineFunc(cumulative.data(), scanOut->scanChunkSums[round][d].data());
            }
        }

        // Scan the elements processed by the CPU serially
        if (scanIn && hostPos < this->dataLength)
        {
            for (uint64_t i = hostPos; i < this->dataLength; i++)
            {
                uint8_t *outElem = scanOut->hostPtr + i * scanOut->elementSize;
                uint8_t *inElem  = scanIn->hostPtr  + i * scanIn->elementSize;
                // scanOut[i] = combine(scanOut[i - 1], in[i])
                std::memcpy(outElem, cumulative.data(), scanOut->elementSize);
                scanOut->combineFunc(outElem, inElem);
                std::memcpy(cumulative.data(), outElem, scanOut->elementSize);
            }
        }
    }
}

// In case the input vector for a reduction is filtered, we need to get the length of the
// inputs vector and make sure there aren't multiple inputs of different sizes
uint64_t *Pipeline::getReduceInputLengthPtrFromArgs(const std::list<KernelArg> &argList)
{
    // If any argument is a filtered vector, it must be the only vector argument
    bool gotFilterArg = false;
    bool gotVectorArg = false;
    uint64_t *lengthPtr = &this->cpuElementCount;
    for (const auto &arg : argList)
    {
        if (arg.isVector())
        {
            if (gotFilterArg)
                return nullptr;
            if (arg.base->isFiltered())
            {
                if (gotVectorArg)
                    return nullptr;
                gotFilterArg = true;
                lengthPtr = &arg.base->cpuLength;
            }
            gotVectorArg = true;
        }
    }
    return lengthPtr;
}

// Check if any of the flags are present in any of the inputs
bool Pipeline::hasInputsWithFlags(const std::list<KernelArg> &kargs, ArgFlags flags)
{
    return std::any_of(kargs.begin(), kargs.end(), [flags](const KernelArg &arg){ return arg.isInput() && (arg.base->flags & flags); });
}

// If we have a window stage that requires overlap, the preceding stage needs to produce more
// output elements than usual in order to have enough inputs for the overlap. The user-provided overlap
// only applies to the very end of the data and not for tasklet boundaries.
void Pipeline::addOverlap(const std::list<Kernel>::reverse_iterator &start, uint8_t *hostPtr, uint32_t overlap)
{
    // Find the most recent stage that produced the arg in question as an output
    for (auto kernel = start; kernel != this->stages.rend(); kernel++)
    {
        for (auto &arg_out : kernel->args)
        {
            if (arg_out.isOutput() && arg_out.base->hostPtr == hostPtr)
            {
                // Make sure that kernel produces enough elements to satisfy the transferOverlap
                kernel->stageOverlap = std::max(kernel->stageOverlap, overlap * kernel->groupSize);
                auto next = std::next(kernel);
                for (auto &arg_in : kernel->args)
                {
                    if (arg_in.isVector())
                    {
                        if (arg_in.isInput())
                            this->addOverlap(next, arg_in.base->hostPtr, kernel->stageOverlap + arg_in.inputOverlap);
                        this->args[arg_in.base->hostPtr].mramOverlap = std::max(this->args[arg_in.base->hostPtr].mramOverlap, kernel->stageOverlap + arg_in.inputOverlap);
                    }
                }
                return;
            }
        }
    }
    this->args[hostPtr].mramOverlap = std::max(this->args[hostPtr].mramOverlap, overlap);
    this->args[hostPtr].transferOverlap = std::max(this->args[hostPtr].transferOverlap, overlap);
}

void Pipeline::timingPoint(const std::string &name, const std::string &reference)
{
#ifdef MEASURE_PERFORMANCE_HOST
    this->timingPoints.emplace_back(name, reference, std::chrono::high_resolution_clock::now());
#endif
}

void Pipeline::emitRecipe()
{
    namespace fs = std::filesystem;

    auto flagsToList = [](const Arg &arg) {
        nlohmann::json a = nlohmann::json::array();
        if (arg.isVector())    a.push_back("Vector");
        if (arg.isScalar())    a.push_back("Scalar");
        if (arg.isInput())     a.push_back("Input");
        if (arg.isOutput())    a.push_back("Output");
        if (arg.isInOut())     a.push_back("InOut");
        if (arg.isFiltered())  a.push_back("Filtered");
        if (arg.isReduced())   a.push_back("Reduced");
        if (arg.isParameter()) a.push_back("Parameter");
        if (arg.isWindowed())  a.push_back("Windowed");
        if (arg.isScanned())   a.push_back("Scanned");
        if (arg.isLookup())    a.push_back("Lookup");
        return a;
    };

    // Order args by index (the map is keyed by host pointer)
    std::vector<const Arg*> orderedArgs;
    for (auto &[ptr, arg] : this->args)
        orderedArgs.push_back(&arg);
    std::sort(orderedArgs.begin(), orderedArgs.end(),
              [](const Arg *a, const Arg *b){ return a->index < b->index; });

    // Element size of the first vector input (informational only)
    uint32_t leadingElementSize = 0;
    for (const Arg *a : orderedArgs)
        if (a->isVector() && a->isInput()) { leadingElementSize = a->elementSize; break; }

    // Count the rounds the same way as execute()
    uint64_t elementsLeftPerDpu = this->totalElementsPerDpu;
    uint32_t numExecutions = 0;
    while (elementsLeftPerDpu > 0)
    {
        uint32_t e = std::min<uint64_t>(elementsLeftPerDpu, this->maxElementsPerDpuPerRound);
        elementsLeftPerDpu -= e;
        numExecutions++;
    }

    nlohmann::json recipe;
    recipe["$schema_version"] = 1;
    recipe["device_source"] = "device.c";
    recipe["pipeline"] = {
        {"data_length",        this->dataLength},
        {"element_size_bytes", leadingElementSize},
        {"dpu_count",          this->dpuCountUsed},
        {"num_tasklets",       Device::taskletCountPerDpu},
        {"num_executions",     numExecutions},
        {"cache_size_bytes",   this->cacheSize},
    };

    recipe["args"] = nlohmann::json::array();
    for (const Arg *arg : orderedArgs)
    {
        // Logical element count. Scalars count as one element per DPU (the WRAM
        // copies are not exposed), filtered outputs report their length before
        // compaction.
        uint64_t logicalLength;
        if (arg->isLookup())
            logicalLength = arg->lookupLength;
        else if (arg->isScalar())
            logicalLength = 1;
        else if (arg->endLengthFraction > 0)
            logicalLength = this->dataLength / arg->endLengthFraction;
        else
            logicalLength = this->dataLength;

        nlohmann::json a;
        a["index"]                 = arg->index;
        a["type"]                  = arg->typeStr;
        a["element_size_bytes"]    = arg->elementSize;
        a["flags"]                 = flagsToList(*arg);
        a["mram_offset"]           = arg->mramOffset;
        a["length_elements"]       = logicalLength;
        a["transfer_overlap"]      = arg->transferOverlap;
        a["mram_overlap_elements"] = arg->mramOverlap;
        if (arg->isFiltered())
            a["filter_length_symbol"] = "filterOutputLengthArg" + std::to_string(arg->index);
        if (arg->isScalar())
            a["wram_symbol"] = "argScalarRaw" + std::to_string(arg->index);
        recipe["args"].push_back(a);
    }

    // Lookup arrays are written once; the consumer must load them into the MRAM
    // of every DPU at mram_offset before the first round.
    recipe["lookups"] = nlohmann::json::array();
    for (const Arg *arg : orderedArgs)
    {
        if (!(arg->isLookup() && arg->isInput() && arg->isPushed()))
            continue;
        fs::create_directories(this->recipeDir + "/inputs/lookups");
        std::string fileRel = "inputs/lookups/arg" + std::to_string(arg->index) + ".bin";
        std::ofstream o(this->recipeDir + "/" + fileRel, std::ios::binary);
        o.write(reinterpret_cast<const char*>(arg->hostPtr),
                static_cast<std::streamsize>(arg->lookupLength) * arg->elementSize);
        recipe["lookups"].push_back({
            {"arg_index",          arg->index},
            {"mram_offset",        arg->mramOffset},
            {"length_elements",    arg->lookupLength},
            {"element_size_bytes", arg->elementSize},
            {"value_bin",          fileRel},
        });
    }

    // Per-round state, following the round loop in execute()
    recipe["executions"] = nlohmann::json::array();
    elementsLeftPerDpu = this->totalElementsPerDpu;
    for (uint32_t round = 0; elementsLeftPerDpu > 0; round++)
    {
        uint32_t elementsPerDpuThisRound = std::min<uint64_t>(elementsLeftPerDpu, this->maxElementsPerDpuPerRound);
        elementsLeftPerDpu -= elementsPerDpuThisRound;
        uint64_t roundBaseArgIndex = static_cast<uint64_t>(round) * this->maxElementsPerDpuPerRound * this->dpuCountUsed;

        nlohmann::json exec;
        exec["index"]                    = round;
        exec["elements_per_dpu"]         = elementsPerDpuThisRound;
        exec["base_arg_offset_elements"] = roundBaseArgIndex;
        exec["host_data"] = {
            {"tasklet_element_count",            elementsPerDpuThisRound / Device::taskletCountPerDpu},
            {"additional_element_tasklet_count", elementsPerDpuThisRound % Device::taskletCountPerDpu},
        };
        exec["inputs"]["mram"]          = nlohmann::json::array();
        exec["inputs"]["wram"]          = nlohmann::json::array();
        exec["outputs"]["mram"]         = nlohmann::json::array();
        exec["outputs"]["wram_symbols"] = nlohmann::json::array();

        std::string execDir = this->recipeDir + "/inputs/exec" + std::to_string(round);

        for (const Arg *arg : orderedArgs)
        {
            // Vector inputs: one file per DPU with its slice of the data
            if (arg->isInput() && arg->isVector() && arg->isPushed())
            {
                nlohmann::json entry;
                entry["arg_index"] = arg->index;
                entry["dpu_files"] = nlohmann::json::array();
                for (uint32_t dpu = 0; dpu < this->dpuCountUsed; dpu++)
                {
                    std::string dpuDir = execDir + "/dpu" + std::to_string(dpu);
                    fs::create_directories(dpuDir);
                    std::string fileRel = "inputs/exec" + std::to_string(round)
                                        + "/dpu" + std::to_string(dpu)
                                        + "/arg" + std::to_string(arg->index) + ".bin";

                    uint64_t srcElem   = roundBaseArgIndex + static_cast<uint64_t>(dpu) * elementsPerDpuThisRound;
                    uint64_t byteCount = (static_cast<uint64_t>(elementsPerDpuThisRound) + arg->transferOverlap) * arg->elementSize;
                    std::ofstream o(this->recipeDir + "/" + fileRel, std::ios::binary);
                    o.write(reinterpret_cast<const char*>(arg->hostPtr) + srcElem * arg->elementSize,
                            byteCount);
                    entry["dpu_files"].push_back(fileRel);
                }
                exec["inputs"]["mram"].push_back(entry);
            }
            // Scalar inputs: one broadcast file, holding R copies for reduce
            // accumulators and a single copy otherwise (must match calcElementCounts)
            if (arg->isInput() && arg->isScalar() && arg->isPushed())
            {
                fs::create_directories(execDir);
                std::string fileRel = "inputs/exec" + std::to_string(round)
                                    + "/argScalarRaw" + std::to_string(arg->index) + ".bin";
                std::ofstream o(this->recipeDir + "/" + fileRel, std::ios::binary);
                uint32_t banks = (arg->isScalar() && arg->isOutput())
                    ? arg->replicationFactor : 1;
                for (uint32_t t = 0; t < banks; t++)
                    o.write(reinterpret_cast<const char*>(arg->hostPtr), arg->elementSize);
                exec["inputs"]["wram"].push_back({
                    {"arg_index", arg->index},
                    {"value_bin", fileRel},
                    {"per_dpu",   false},
                });
            }
            // Outputs to read back
            if (arg->isOutput() && arg->isFetched() && arg->isVector())
                exec["outputs"]["mram"].push_back({{"arg_index", arg->index}});
            if (arg->isOutput() && arg->isFetched() && arg->isFiltered())
                exec["outputs"]["wram_symbols"].push_back("filterOutputLengthArg" + std::to_string(arg->index));
        }

        recipe["executions"].push_back(exec);
    }

    // Host post-processing information (only used by the cost model)
    nlohmann::json post;
    post["cpu_worker_element_count"] = this->cpuElementCount;
    post["compress_filtered"]        = nlohmann::json::array();
    post["combine_reduce"]           = nlohmann::json::array();
    for (const Arg *arg : orderedArgs)
    {
        if (arg->isFiltered() && arg->isOutput())
            post["compress_filtered"].push_back({
                {"arg_index",     arg->index},
                {"length_symbol", "filterOutputLengthArg" + std::to_string(arg->index)},
            });
        if (arg->isReduced() && arg->isOutput())
            post["combine_reduce"].push_back({{"arg_index", arg->index}});
    }
    recipe["host_post_processing"] = post;

    std::ofstream out(this->recipeDir + "/recipe.json");
    out << recipe.dump(2);
}

// Processing-using-DRAM backend: lowers the MAP / REDUCE / GROUP stages to a bbop
// harness for the SIMDRAM analytical cost model (bbop_manager.c). A GROUP stage
// (N independent reductions of length G) is mapped either output-stationary (the
// N outputs are packed across the SIMD lanes and the reduction becomes G sequential
// multiply-accumulate steps) or reduction-in-lane (one length-G reduction per
// output, as SIMDRAM's inner-loop vectorization does), whichever is cheaper.
// Operands are assumed to be square matrices.
void Pipeline::emitBbop()
{
    namespace fs = std::filesystem;
    fs::create_directories(this->bbopDir);

    struct StageOp {
        int prim;                       // 0=MAP, 1=REDUCE, 2=GROUP
        uint64_t nOut;                  // independent outputs (SIMD-packed lanes)
        uint64_t g;                     // contraction length (sequential MAC steps)
        std::vector<uint32_t> operands; // arg indices used as MUL operands (>= 1)
        int outIndex;                   // output arg index (-1 if none)
    };
    std::vector<StageOp> ops;
    std::map<uint32_t, uint64_t> argSize;   // arg index -> element count

    for (Kernel &k : this->stages)
    {
        StageOp op;
        if (k.isReduce())        { op.prim = 1; op.nOut = 1;                 op.g = this->dataLength; }
        else if (k.groupSize > 1){ op.prim = 2; op.nOut = this->dataLength / k.groupSize; op.g = k.groupSize; }
        else                     { op.prim = 0; op.nOut = this->dataLength;  op.g = 1; }

        op.outIndex = -1;
        for (KernelArg &ka : k.args)
        {
            uint32_t idx = ka.base->index;
            if (ka.isVector() && (ka.isInput() || ka.isOutput()))
                argSize[idx] = std::max<uint64_t>(argSize[idx], op.nOut);
            else if (ka.isParameter())
                argSize[idx] = std::max<uint64_t>(argSize[idx], op.nOut);
            if (ka.isVector() && ka.isInput())    op.operands.push_back(idx);
            else if (ka.isParameter())            op.operands.push_back(idx);
            if (ka.isVector() && ka.isOutput())   op.outIndex = static_cast<int>(idx);
        }
        ops.push_back(op);
    }

    // Generate the harness
    std::string h;
    h += "/* Auto-generated by DaPPA PUD backend (DAPPA_EMIT_BBOP). */\n";
    h += "#include <stdlib.h>\n#include <stdio.h>\n";
    h += "#include \"bbop_manager.h\"\n\n";
    h += "int main(void){\n";
    h += "    initialize_bbop_statistics();\n";
    h += "    const long SCR = 65536;\n";

    // One distinct buffer per arg. The cost model only uses the pointer values and
    // sizes, and does not touch the data when built with BBOP_SKIP_COMPUTE.
    for (auto &[ptr, arg] : this->args)
        h += "    DATATYPE_BBOP *buf" + std::to_string(arg.index)
           + " = (DATATYPE_BBOP*)malloc(sizeof(DATATYPE_BBOP)*SCR);\n";
    h += "    DATATYPE_BBOP *prod = (DATATYPE_BBOP*)malloc(sizeof(DATATYPE_BBOP)*SCR);\n";
    h += "    DATATYPE_BBOP *acc  = (DATATYPE_BBOP*)malloc(sizeof(DATATYPE_BBOP)*SCR);\n\n";

    // Register the operands that need to be transposed to the vertical layout.
    // Scalar parameters are replicated across all lanes and transposed as well.
    for (auto &[ptr, arg] : this->args)
    {
        uint64_t sz = argSize.count(arg.index) ? argSize[arg.index] : this->dataLength;
        std::string b = "buf" + std::to_string(arg.index);
        std::string s = std::to_string(sz);
        if (arg.isVector())
        {
            if (arg.isInput())  h += "    bbop_register_input("  + b + ", " + s + ");\n";
            if (arg.isOutput()) h += "    bbop_register_output(" + b + ", " + s + ");\n";
        }
        else if (arg.isParameter())
            h += "    bbop_register_input(" + b + ", " + s + ");\n";
    }
    h += "\n";

    int id = 0;
    for (size_t si = 0; si < ops.size(); si++)
    {
        StageOp &op = ops[si];
        std::string N = std::to_string(op.nOut);
        std::string opA = op.operands.size() > 0 ? "buf" + std::to_string(op.operands[0]) : "acc";
        std::string opB = op.operands.size() > 1 ? "buf" + std::to_string(op.operands[1]) : opA;
        std::string outBuf = op.outIndex >= 0 ? "buf" + std::to_string(op.outIndex) : "acc";

        h += "    /* stage " + std::to_string(si) + ": prim=" + std::to_string(op.prim)
           + " N=" + N + " G=" + std::to_string(op.g) + " */\n";
        if (op.prim == 1)
        {
            // Reduction to a scalar (free in the cost model)
            h += "    " + outBuf + "[0] = bbop_op_red(BBOP_ADD, " + opA + ", "
               + std::to_string(op.g) + ", " + std::to_string(id++) + ");\n";
        }
        else if (op.prim == 0)
        {
            // Element-wise map
            h += "    bbop_op(BBOP_MUL, " + opA + ", " + opB + ", " + outBuf + ", "
               + N + ", " + std::to_string(id++) + ", MUL_SSL_Sklansky_24);\n";
        }
        else
        {
            // Group: choose the mapping with fewer multiplication passes.
            //   Output-stationary: g * ceil(nOut / 64K) passes
            //   Reduction-in-lane: nOut * ceil(g / 64K) passes
            // Output-stationary wins when there are more outputs than reduction
            // steps (e.g., GEMM); otherwise (e.g., GEMV) reduction-in-lane is used.
            const uint64_t W = 65536;
            uint64_t osCost  = op.g    * ((op.nOut + W - 1) / W);
            uint64_t rilCost = op.nOut * ((op.g    + W - 1) / W);
            if (osCost <= rilCost)
            {
                // Output-stationary
                int mulId = id++, addId = id++;
                h += "    for (long k = 0; k < " + std::to_string(op.g) + "; k++) {\n";
                h += "        bbop_op(BBOP_MUL, " + opA + ", " + opB + ", prod, " + N + ", "
                   + std::to_string(mulId) + ", MUL_SSL_Sklansky_24);\n";
                h += "        bbop_op(BBOP_ADD, acc, prod, " + outBuf + ", " + N + ", "
                   + std::to_string(addId) + ", BBOP_ADD);\n";
                h += "    }\n";
            }
            else
            {
                // Reduction-in-lane. The result of bbop_op_red is discarded, as
                // nOut may exceed the size of the scratch buffers.
                int mulId = id++, redId = id++;
                h += "    for (long i = 0; i < " + N + "; i++) {\n";
                h += "        bbop_op(BBOP_MUL, " + opA + ", " + opB + ", prod, "
                   + std::to_string(op.g) + ", " + std::to_string(mulId)
                   + ", MUL_SSL_Sklansky_24);\n";
                h += "        bbop_op_red(BBOP_ADD, prod, " + std::to_string(op.g) + ", "
                   + std::to_string(redId) + ");\n";
                h += "    }\n";
            }
        }
    }

    h += "\n    print_bbop_statistic();\n    return 0;\n}\n";

    std::ofstream(this->bbopDir + "/harness.c") << h;
}

}
