#ifndef DAPPA_PIPELINE_FULL_H
#define DAPPA_PIPELINE_FULL_H

#include <list>
#include "pipeline.h"

namespace DaPPA
{

class PipelineFull
{
public:
    explicit PipelineFull(uint64_t _dataLength);
    void addHeader(const std::string &data);
    bool execute();
    void fetch(void *ptr);
    uint64_t getLength(void *ptr);

    template<typename T>
    void fetch(std::vector<T> &vec){ this->fetch(vec.data()); }

    template<typename T>
    uint64_t getLength(std::vector<T> &vec){ return this->getLength(vec.data()); }

    // Map / Window / Group
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
    {
        if (this->hasFailed)
            return false;
        if (this->needsNewStage(argsTuple, ArgFlags::Filtered | ArgFlags::Reduced | ArgFlags::Scanned))
            this->pipelineParts.emplace_back(this->dataLength);
        return !(this->hasFailed = this->hasFailed || !this->pipelineParts.back().stage(str, func, argsTuple, overlap, groupSize));
    }

    // Filter (Input/Output)
    template<typename FilterType, typename... ArgTypes>
    bool stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> output, std::tuple<ArgTyped<FilterType>, ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
    {
        if (this->hasFailed)
            return false;
        if (this->needsNewStage(std::tuple_cat(argsTuple, std::make_tuple(output)), ArgFlags::Reduced | ArgFlags::Scanned))
            this->pipelineParts.emplace_back(this->dataLength);
        return !(this->hasFailed = this->hasFailed || !this->pipelineParts.back().stage(str, func, output, argsTuple, overlap, groupSize));
    }

    // Filter (InOut)
    template<typename FilterType, typename... ArgTypes>
    bool stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> inout, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
    {
        if (this->hasFailed)
            return false;
        if (this->needsNewStage(std::tuple_cat(argsTuple, std::make_tuple(inout)), ArgFlags::Reduced | ArgFlags::Scanned))
            this->pipelineParts.emplace_back(this->dataLength);
        return !(this->hasFailed = this->hasFailed || !this->pipelineParts.back().stage(str, func, inout, argsTuple, overlap, groupSize));
    }

    // Reduce
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple)
    {
        if (this->hasFailed)
            return false;
        if (this->needsNewStage(argsTuple, ArgFlags::Reduced | ArgFlags::Scanned))
            this->pipelineParts.emplace_back(this->dataLength);
        return !(this->hasFailed = this->hasFailed || !this->pipelineParts.back().stage(str, func, argsTuple));
    }

    // Scan
    template<typename... ArgTypes>
    bool stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple, ScanTag)
    {
        if (this->hasFailed)
            return false;
        if (this->needsNewStage(argsTuple, ArgFlags::Reduced | ArgFlags::Scanned))
            this->pipelineParts.emplace_back(this->dataLength);
        return !(this->hasFailed = this->hasFailed || !this->pipelineParts.back().stage(str, func, argsTuple, ScanTag{}));
    }

private:
    uint64_t dataLength;
    std::list<Pipeline> pipelineParts;
    std::list<std::string> headers;
    bool hasFailed;

    template<typename... ArgTypes>
    bool needsNewStage(const std::tuple<ArgTyped<ArgTypes>...> &argsTuple, ArgFlags flags)
    {
        for (const auto &part : this->pipelineParts)
        {
            for (const auto &[ptr, arg] : part.args)
            {
                auto &oldArg = arg;
                if (std::apply([oldArg, flags](const auto &... newArg){ return ((newArg.hostPtr == oldArg.hostPtr && (oldArg.flags & flags)) || ...); }, argsTuple))
                    return true;
            }
        }
        return false;
    }
};

}

#endif //DAPPA_PIPELINE_FULL_H
