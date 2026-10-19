#ifndef DAPPA_ARGTYPED_H
#define DAPPA_ARGTYPED_H

#include <vector>
#include "arg.h"
#include "arg_flags.h"
#include "device.h"

namespace DaPPA
{

template<typename T>
class ArgTyped : public Arg
{
public:
    ArgTyped(const std::string &_typeStr, T *_hostPtrTyped, ArgFlags _flags, const T &padding)
            : Arg(_flags), paddingSize(0), paddingValue(padding), paddingPtr(nullptr)
    { this->init(_typeStr, _hostPtrTyped, {}); }

    ArgTyped(const std::string &_typeStr, T *_hostPtrTyped, ArgFlags _flags, T *padding, uint32_t parameterSize = 0)
            : Arg(_flags), paddingSize(0), paddingPtr(padding)
    { this->init(_typeStr, _hostPtrTyped, {}, parameterSize); }

    ArgTyped(const std::string &_typeStr, T *_hostPtrTyped, ArgFlags _flags, std::vector<T> padding)
            : Arg(_flags), paddingSize(padding.size()), paddingPtr(padding.data())
    { this->init(_typeStr, _hostPtrTyped, {}); }

    ArgTyped(const std::string &_typeStr, T *_hostPtrTyped, ArgFlags _flags, std::function<void(T *, const T *)> _combineFuncTyped = {}, uint32_t parameterSize = 0)
            : Arg(_flags), paddingSize(0), paddingPtr(nullptr)
    { this->init(_typeStr, _hostPtrTyped, _combineFuncTyped, parameterSize); }

    ArgTyped(const std::string &_typeStr, std::vector<T> &hostVectorTyped, ArgFlags _flags, const T &padding)
            : Arg(_flags), paddingSize(0), paddingValue(padding), paddingPtr(nullptr)
    { this->init(_typeStr, hostVectorTyped.data(), {}); }

    ArgTyped(const std::string &_typeStr, std::vector<T> &hostVectorTyped, ArgFlags _flags, T *padding, uint32_t parameterSize = 0)
            : Arg(_flags), paddingSize(0), paddingPtr(padding)
    { this->init(_typeStr, hostVectorTyped.data(), {}, parameterSize); }

    ArgTyped(const std::string &_typeStr, std::vector<T> &hostVectorTyped, ArgFlags _flags, std::vector<T> padding)
            : Arg(_flags), paddingSize(padding.size()), paddingPtr(padding.data())
    { this->init(_typeStr, hostVectorTyped.data(), {}); }

    ArgTyped(const std::string &_typeStr, std::vector<T> &hostVectorTyped, ArgFlags _flags, std::function<void(T *, const T *)> _combineFuncTyped = {}, uint32_t parameterSize = 0)
            : Arg(_flags), paddingSize(0), paddingPtr(nullptr)
    { this->init(_typeStr, hostVectorTyped.data(), _combineFuncTyped, parameterSize); }

    using Type = T;

    T paddingValue;
    T *paddingPtr;
    uint32_t paddingSize;

private:
    void init(const std::string &_typeStr, T *_hostPtrTyped, std::function<void(T *, const T *)> _combineFuncTyped, uint32_t parameterSize = 0)
    {
        this->typeStr = _typeStr;
        this->hostPtr = reinterpret_cast<uint8_t*>(_hostPtrTyped);

        if (this->isScalar() && parameterSize != 0)
            this->elementSize = Device::paddedSizeWram(parameterSize);
        else if (this->isScalar())
            this->elementSize = Device::paddedSizeWram(sizeof(T));
        else
            this->elementSize = sizeof(T);

        // Lookup tables pass their total size in bytes via parameterSize
        this->lookupLength = this->isLookup() ? (parameterSize / sizeof(T)) : 0;

        if (_combineFuncTyped)
        {
            this->combineFunc = [func=_combineFuncTyped](void *dst, const void *src){
                func(reinterpret_cast<T*>(dst), reinterpret_cast<const T*>(src));
            };
        }
    }
};

}

#endif // DAPPA_ARGTYPED_H
