#ifndef DAPPA_PIPELINE_H
#define DAPPA_PIPELINE_H

#include <list>
#include <map>
#include <chrono>
#include <utility>
#include "nlohmann/json.hpp"
#include "kernel.h"
#include "arg.h"
#include "arg_typed.h"
#include "logging.h"

namespace DaPPA {

// Tag appended by the SCAN macro to distinguish its stage() overload from REDUCE.
struct ScanTag {};

class Pipeline
{
public:
    explicit Pipeline(uint64_t _dataLength);
    ~Pipeline();
    void addHeader(const std::string &data);
    bool execute();
    void fetch(void *ptr, bool silent = false);
    uint64_t getLength(void *ptr);

    template<typename T>
    void fetch(std::vector<T> &vec, bool silent = false){ this->fetch(vec.data(), silent); }

    template<typename T>
    uint64_t getLength(std::vector<T> &vec){ return this->getLength(vec.data()); }

    // Map / Window / Group
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize);

    // Filter (Input/Output)
    template<typename FilterType, typename... ArgTypes>
    bool stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> output, std::tuple<ArgTyped<FilterType>, ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize);

    // Filter (InOut)
    template<typename FilterType, typename... ArgTypes>
    bool stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> inout, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize);

    // Reduce
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple);

    // Scan (inclusive prefix fold). The combiner has the same shape as for Reduce;
    // the args are an INOUT(T, scanOut) vector initialized with the identity
    // element and an INPUT(T, in) vector.
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple, ScanTag);

private:
    uint64_t dataLength;
    bool hasFailed;
    uint32_t dpuCountUsed;
    uint64_t totalElementsPerDpu;
    uint64_t cpuElementCount;
    uint64_t totalElementsPerTasklet;
    uint32_t taskletMaxElementsPerRound;
    uint32_t maxElementsPerDpuPerRound;
    uint32_t maxGroupSize;
    uint32_t segmentLength;
    uint32_t cacheSize;
    std::map<uint8_t*, Arg> args;
    std::list<Kernel> stages;
    std::list<std::string> headers;
    std::string recipeDir;
    // Processing-using-DRAM backend: DAPPA_EMIT_BBOP=<dir> emits a bbop harness
    // for the SIMDRAM cost model instead of running on UPMEM.
    std::string bbopDir;
#ifdef MEASURE_PERFORMANCE_HOST
    struct TimingPoint
    {
        TimingPoint(std::string _name, std::string _reference, std::chrono::high_resolution_clock::time_point _timePoint)
        : name(std::move(_name)), reference(std::move(_reference)), timePoint(_timePoint){};
        std::string name;
        std::string reference;
        std::chrono::high_resolution_clock::time_point timePoint;
    };
    std::list<TimingPoint> timingPoints;
#endif

    template<typename... ArgTypes>
    std::list<KernelArg> addArgs(std::tuple<ArgTyped<ArgTypes>...> &argsTuple, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction);

    template<typename FilterType, typename... ArgTypes>
    std::list<KernelArg> addArgs(std::tuple<ArgTyped<ArgTypes>...> &argsTuple, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction, ArgTyped<FilterType> &filterOutput);

    template<typename ArgType>
    KernelArg addArg(const ArgTyped<ArgType> &arg, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction);

    template<typename... ArgTypes>
    uint32_t getInputLengthFraction(std::tuple<ArgTyped<ArgTypes>...> &argsTuple);

    void calcOffsets();
    void calcElementCounts();
    void compressFilteredCpuArgs();
    void combineReductions();
    // Apply the cross-round and cross-DPU offsets to every scanned output
    void combineScans();
    nlohmann::json buildJson();
    void cpuWorker();
    uint64_t *getReduceInputLengthPtrFromArgs(const std::list<KernelArg> &argList);
    void addOverlap(const std::list<Kernel>::reverse_iterator &start, uint8_t *hostPtr, uint32_t overlap);

    static std::string createTempDir(uint32_t maxTries = 1000);
    [[nodiscard]] static bool hasInputsWithFlags(const std::list<KernelArg> &kargs, ArgFlags flags);
    void timingPoint(const std::string &name = "", const std::string &reference = "");
    void emitRecipe();
    void emitBbop();

    friend class PipelineFull;
};

}

// Helper macros for main header
#define DAPPA_FILTER_EXPAND(arg, ...) arg, std::make_tuple(__VA_ARGS__)

#include "pipeline.tpp"

#endif //DAPPA_PIPELINE_H
