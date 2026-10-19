#include "kernel_arg.h"

namespace DaPPA
{

KernelArg::KernelArg()
        :
        ArgFlagsBase(ArgFlags::Invalid),
        base(nullptr),
        cacheOffset(0),
        inputOverlap(0)
{
}

KernelArg::KernelArg(Arg *_arg, const ArgFlagsBase &_flags, uint32_t overlap, const void *paddingPtr, bool isPaddingPtr)
        :
        ArgFlagsBase(_flags),
        base(_arg),
        cacheOffset(0),
        inputOverlap(0)
{
    // Overlap only applies to the inputs of a stage
    if (this->isInput())
        this->inputOverlap = overlap;

    if (overlap > 0 && this->isInput())
    {
        // Create and fill buffers to store the overlap data in that will be appended to
        // the very end of the data
        this->paddingBuffer.resize(this->base->elementSize * overlap);
        if (isPaddingPtr)
        {
            std::memcpy(this->paddingBuffer.data(), paddingPtr, this->base->elementSize * overlap);
        }
        else
        {
            for (uint32_t i = 0; i < overlap; i++)
                std::memcpy(this->paddingBuffer.data() + (i * this->base->elementSize), paddingPtr, this->base->elementSize);
        }
    }
}

// Produces a JSON object of the current arg that can be used in the DPU code template
nlohmann::json KernelArg::toJson()
{
    if (this->json.empty())
    {
        this->json = this->base->toJson();
        this->json["cacheOffset"] = this->cacheOffset;
        this->json["inputOverlap"] = this->inputOverlap;

        // Override the flags as they might change between stages
        this->json["isInput"] = this->isInput();
        this->json["isOutput"] = this->isOutput();
        this->json["isScalar"] = this->isScalar();
        this->json["isVector"] = this->isVector();
        this->json["isFiltered"] = this->isFiltered();
        this->json["isWindowed"] = this->isWindowed();
    }

    return this->json;
}

}
