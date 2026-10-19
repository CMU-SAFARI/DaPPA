#include "arg_flags.h"

namespace DaPPA
{

ArgFlagsBase::ArgFlagsBase(ArgFlags flags)
        :
        flags(flags)
{
}

bool ArgFlagsBase::isValid() const
{
    return uint32_t(this->flags) > 0;
}

bool ArgFlagsBase::isInput() const
{
    return this->flags & ArgFlags::Input;
}

bool ArgFlagsBase::isOutput() const
{
    return this->flags & ArgFlags::Output;
}

bool ArgFlagsBase::isInOut() const
{
    return (this->flags & ArgFlags::Input) && (this->flags & ArgFlags::Output);
}

bool ArgFlagsBase::isVector() const
{
    return this->flags & ArgFlags::Vector;
}

bool ArgFlagsBase::isScalar() const
{
    return this->flags & ArgFlags::Scalar;
}

bool ArgFlagsBase::isFiltered() const
{
    return this->flags & ArgFlags::Filtered;
}

bool ArgFlagsBase::isFetched() const
{
    return this->flags & ArgFlags::Fetched;
}

bool ArgFlagsBase::isPushed() const
{
    return this->flags & ArgFlags::Pushed;
}

bool ArgFlagsBase::isWindowed() const
{
    return this->flags & ArgFlags::Windowed;
}

bool ArgFlagsBase::isReduced() const
{
    return this->flags & ArgFlags::Reduced;;
}

bool ArgFlagsBase::isParameter() const
{
    return this->flags & ArgFlags::Parameter;;
}

bool ArgFlagsBase::isScanned() const
{
    return this->flags & ArgFlags::Scanned;
}

bool ArgFlagsBase::isLookup() const
{
    return this->flags & ArgFlags::Lookup;
}

void ArgFlagsBase::markFiltered()
{
    this->flags = this->flags | ArgFlags::Filtered;
}

void ArgFlagsBase::markFetched()
{
    this->flags = this->flags | ArgFlags::Fetched;
}

void ArgFlagsBase::markPushed()
{
    this->flags = this->flags | ArgFlags::Pushed;
}

void ArgFlagsBase::markWindowed()
{
    this->flags = this->flags | ArgFlags::Windowed;
}

void ArgFlagsBase::markGrouped()
{
    this->flags = this->flags | ArgFlags::Grouped;
}

void ArgFlagsBase::markScanned()
{
    this->flags = this->flags | ArgFlags::Scanned;
}

}
