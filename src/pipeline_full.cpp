#include "pipeline_full.h"

namespace DaPPA
{

PipelineFull::PipelineFull(uint64_t _dataLength)
        :
        dataLength(_dataLength),
        hasFailed(false)
{
    this->pipelineParts.emplace_back(dataLength);
}

void PipelineFull::addHeader(const std::string &data)
{
    this->headers.emplace_back(data);
}

void PipelineFull::fetch(void *ptr)
{
    for (auto &part : this->pipelineParts)
        part.fetch(ptr, true);
}

uint64_t PipelineFull::getLength(void *ptr)
{
    for (auto part = this->pipelineParts.rbegin(); part != this->pipelineParts.rend(); part++)
    {
        uint64_t length = part->getLength(ptr);
        if (length != 0)
            return length;
    }
    return 0;
}

bool PipelineFull::execute()
{
    if (this->hasFailed)
        return false;

    for (auto &part : this->pipelineParts)
        for (auto &header : this->headers)
            part.addHeader(header);

    for (auto part = this->pipelineParts.begin(); part != this->pipelineParts.end(); part++)
    {
        // Search for input data that might reduce the length of the calculations
        uint64_t inputLength = this->dataLength;
        for (auto rpart = std::make_reverse_iterator(part); rpart != this->pipelineParts.rend(); rpart++)
            for (auto &[ptr, arg] : rpart->args)
                if (part->args.find(arg.hostPtr) != part->args.end() && arg.isFiltered())
                    inputLength = std::min(inputLength, rpart->getLength(arg.hostPtr));

        // Search for output data that will be required for succeeding stages and fetch it
        for (auto next = std::next(part); next != this->pipelineParts.end(); next++)
            for (auto &[nextPtr, nextArg] : next->args)
                for (auto &[curPtr, curArg] : part->args)
                    if (curArg.hostPtr == nextArg.hostPtr)
                        part->fetch(curArg.hostPtr);

        part->dataLength = inputLength;
        if (!part->execute())
            return false;
    }

    return true;
}

}
