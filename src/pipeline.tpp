#include <iostream>

namespace DaPPA
{

// Stores type information along with a pointer to an argument
template<typename T>
class TypeWrapper
{
public:
    explicit TypeWrapper(const ArgTyped<T> &arg, Arg *_value) : value(_value), flags(arg){}

    Arg* operator ->() const { return this->value; }
    T *cast(void *ptr) const { return reinterpret_cast<T*>(ptr); }

    Arg *value;
    ArgFlagsBase flags;
};

// Add a new arg to the global argument list or update it if it already exists
template<typename ArgType>
KernelArg Pipeline::addArg(const ArgTyped<ArgType> &arg, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction)
{
    if (arg.paddingSize != 0 && arg.paddingSize != overlap)
    {
        this->hasFailed = true;
        ERROR_HOST("Supplied padding of incorrect size (expected %d, got %d)", overlap, arg.paddingSize);
        return {};
    }

    // Search for the new argument in the global argument list
    auto *ptr = reinterpret_cast<uint8_t*>(arg.hostPtr);
    if (this->args.find(ptr) == this->args.end())
    {
        if (arg.isScalar() && arg.isOutput() && !arg.combineFunc)
        {
            this->hasFailed = true;
            ERROR_HOST("Missing combine function for reduction");
            return {};
        }

        // If the argument was not found, we need to construct a new typeless arg and append it
        // to the global list
        auto index = this->args.size();
        this->args[ptr] = arg;

        if (arg.isVector() && arg.isOutput() && !arg.isInput())
            this->args[ptr].init(index, inputLengthFraction * groupSize, this->dpuCountUsed);
        else
            this->args[ptr].init(index, inputLengthFraction, this->dpuCountUsed);

        // Only mark it as pushed if it first appears as an input
        if (arg.isInput())
            this->args[ptr].markPushed();
    }

    // Make sure the arg in the global list has all necessary flags
    auto *argPtr = &this->args[ptr];
    argPtr->flags = argPtr->flags | arg.flags;

    // Make sure preceding stages are aware that they may have to produce more output
    // in order to satisfy the inputOverlap
    if (arg.isVector() && arg.isInput() && overlap > 0)
    {
        this->addOverlap(this->stages.rbegin(), arg.hostPtr, overlap);
        argPtr->maxPadding = std::max(arg.maxPadding, overlap);
    }

    // Fix the length fractions after group stages
    if (arg.isOutput())
    {
        argPtr->endLengthFraction = inputLengthFraction * groupSize;
        if (groupSize > 1)
            argPtr->markGrouped();
        if (overlap > 0)
            argPtr->markWindowed();
    }
    else
    {
        argPtr->endLengthFraction = inputLengthFraction;
    }

    // Make sure inputs that have been filtered in previous stages are marked as such
    ArgFlagsBase flagsBase = arg;
    if (argPtr->isFiltered())
        flagsBase.markFiltered();

    if (arg.paddingPtr)
        return {argPtr, flagsBase, overlap, arg.paddingPtr, true};
    else
        return {argPtr, flagsBase, overlap, &arg.paddingValue, false};
}

// Add a set of new args to the global argument list or update them if they already exist
template<typename... ArgTypes>
std::list<KernelArg> Pipeline::addArgs(std::tuple<ArgTyped<ArgTypes>...> &argsTuple, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction)
{
    std::list<KernelArg> argList;

    // Remove the types from the args and link them to the proper argument in the args list:
    // std::apply unpacks the tuple, a comma fold expression adds each arg to the global list
    std::apply(
        [&argList, overlap, groupSize, inputLengthFraction, this](auto &... arg){
            ([&argList, overlap, groupSize, inputLengthFraction, this](auto &arg){
                KernelArg karg = this->addArg(arg, overlap, groupSize, inputLengthFraction);
                if (!karg.isValid())
                {
                    argList.clear();
                    return;
                }
                argList.push_back(karg);
            }(arg), ...);
        }, argsTuple);

    return argList;
}

// Add a set of new args to the global argument list or update them if they already exist (filter version)
template<typename FilterType, typename... ArgTypes>
std::list<KernelArg> Pipeline::addArgs(std::tuple<ArgTyped<ArgTypes>...> &argsTuple, uint32_t overlap, uint32_t groupSize, uint32_t inputLengthFraction, ArgTyped<FilterType> &filterOutput)
{
    auto argList = this->addArgs(argsTuple, overlap, groupSize, inputLengthFraction);

    if (filterOutput.isValid())
    {
        // Add the filter output as an arg as it is not included in the regular argument list
        filterOutput.markFiltered();
        KernelArg karg = this->addArg(filterOutput, overlap, groupSize, inputLengthFraction);
        if (!karg.isValid())
            return {};

        // The base only appears in the argument list the kernel sees for InOut types, not for Output types
        if (filterOutput.isInOut())
            argList.push_back(karg);
    }

    return argList;
}

// Get the common length fraction for the inputs of a stage
template<typename... ArgTypes>
uint32_t Pipeline::getInputLengthFraction(std::tuple<ArgTyped<ArgTypes>...> &argsTuple)
{
    // std::apply unpacks the tuple and a comma fold expression visits each argument:
    //  - filtered arguments always have a fraction of 1, as their length is tracked separately
    //  - arguments not in the global list are skipped
    //  - a fraction that does not match the previous ones invalidates the result
    uint32_t fraction = std::apply([this](const auto &... arg){
        uint32_t fraction = 0;
        uint32_t newFraction = 1;
        ((arg.isVector()
          && arg.isInput()
          && (newFraction = arg.isFiltered()
                  ? 1
                  : this->args.find(arg.hostPtr) == this->args.end()
                  ? 0
                  : this->args[arg.hostPtr].endLengthFraction,
              fraction = fraction == 0 || fraction == newFraction
                  ? newFraction
                  : -1)
         ), ...);
        return fraction;
    }, argsTuple);

    if (fraction == -1)
    {
        this->hasFailed = true;
        ERROR_HOST("Not all inputs share the same group size");
        return -1;
    }
    return fraction == 0 ? 1 : fraction;
}

// Map / Window / Group
template<typename... ArgTypes>
bool Pipeline::stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
{
    DEBUG_HOST_V("Adding map/window/group stage");
    this->timingPoint();

    // Get the input length fraction in case the arguments won't occupy the whole space
    uint32_t inputLengthFraction = this->getInputLengthFraction(argsTuple);
    if (inputLengthFraction == -1)
        return false;

    // Remove the types from the args and link them to the proper argument in the args list
    auto argList = this->addArgs(argsTuple, overlap, groupSize, inputLengthFraction);
    if (argList.empty())
        return false;

    // Make sure the overlap size is valid
    if (this->dataLength < overlap + 1)
    {
        this->hasFailed = true;
        ERROR_HOST("Input too small to fill at least one window");
        return false;
    }

    // Check input flags
    if (hasInputsWithFlags(argList, ArgFlags::Filtered | ArgFlags::Reduced))
    {
        this->hasFailed = true;
        ERROR_HOST("Map/window/group operation may not have filtered or reduced inputs");
        return false;
    }

    // Check for inouts for group/window
    if ((groupSize > 1 || overlap > 0) && hasInputsWithFlags(argList, ArgFlags::InOut))
    {
        this->hasFailed = true;
        ERROR_HOST("Window/group operation may not have inouts");
        return false;
    }

    // Update the global max group size
    this->maxGroupSize = std::max(this->maxGroupSize, groupSize);

    // The typed args tuple won't reflect changes made to the arguments from within the pipeline, so we need
    // to construct a tuple of pointers instead
    auto argPtrsTuple = std::apply(
        [this](auto &... arg){
            return std::make_tuple(TypeWrapper(arg, &this->args[arg.hostPtr])...);
        }, argsTuple);

    // Create a function that will process an element on the CPU
    std::function cpuFunc = [this, func, argPtrsTuple](const Kernel *kernel){
        std::apply([this, kernel, func](auto &&... arg) {
            // Copy the padding buffer for the current stage
            for (const auto &karg : kernel->args)
                if (karg.isVector() && karg.isInput() && karg.inputOverlap > 0)
                    std::memcpy(karg.base->cpuPtr + (this->cpuElementCount / kernel->inputLengthFraction) * karg.base->elementSize,
                                karg.paddingBuffer.data(), karg.inputOverlap * karg.base->elementSize);

            for (uint64_t i = 0; i < this->cpuElementCount / kernel->outputLengthFraction; i++)
                func((arg.cast(arg->cpuPtr) + (arg->isVector() ? (arg.flags.isOutput() ? i : i * kernel->groupSize) : 0))...);
        }, argPtrsTuple);
    };

    // Store the kernel information
    this->stages.emplace_back(str, cpuFunc, groupSize, inputLengthFraction, KernelArg(), argList);

    this->timingPoint("New stage (map/window/group)");

    return true;
}

// Filter (Input/Output)
template<typename FilterType, typename... ArgTypes>
bool Pipeline::stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> output, std::tuple<ArgTyped<FilterType>, ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
{
    DEBUG_HOST_V("Adding filter input/output stage");
    this->timingPoint();

    // Get the input length fraction in case the arguments won't occupy the whole space
    uint32_t inputLengthFraction = this->getInputLengthFraction(argsTuple);

    // Do the operations common to all stage types
    auto argList = this->addArgs(argsTuple, overlap, groupSize, inputLengthFraction, output);
    if (argList.empty())
        return false;

    // Check input flags
    if (hasInputsWithFlags(argList, ArgFlags::Reduced))
    {
        this->hasFailed = true;
        ERROR_HOST("Filter operation may not have reduced inputs");
        return false;
    }

    // Get the input from the tuple
    auto input = std::get<0>(argsTuple);

    // Ensure both arguments have the correct IO type
    if (!output.isOutput() || !input.isInput() || output.isInOut())
    {
        this->hasFailed = true;
        ERROR_HOST("Filter input/output has wrong I/O type");
        return false;
    }

    // The typed args tuple won't reflect changes made to the arguments from within the pipeline, so we need
    // to construct a tuple of pointers instead
    auto argPtrsTuple = std::apply(
        [this](auto &... arg){
            return std::make_tuple(TypeWrapper(arg, &this->args[arg.hostPtr])...);
        }, argsTuple);

    // Get typed pointers for the input/output for the same reason as the tuple above
    auto inputPtr = TypeWrapper(input, &this->args[input.hostPtr]);
    auto outputPtr = TypeWrapper(output, &this->args[output.hostPtr]);

    // Create a function that will process the CPU elements
    auto cpuFunc = [outputPtr, inputPtr, func, argPtrsTuple](const Kernel *kernel) {
        std::apply([outputPtr, inputPtr, func, kernel](auto &&... arg) {
            // Copy the padding buffer for the current stage
            for (const auto &karg : kernel->args)
                if (karg.isInput() && karg.inputOverlap > 0)
                    std::memcpy(karg.base->cpuPtr + inputPtr->cpuLength * karg.base->elementSize / kernel->inputLengthFraction,
                                karg.paddingBuffer.data(), karg.inputOverlap * karg.base->elementSize);

            uint64_t writeIndex = 0;
            for (uint64_t i = 0; i < inputPtr->cpuLength / kernel->groupSize; i++)
            {
                if (func((arg.cast(arg->cpuPtr) + (arg->isVector() ? (arg.flags.isOutput() ? i : i * kernel->groupSize) : 0))...))
                {
                    DEBUG_HOST_V("Writing filtered output: %lu -> %lu", i, writeIndex);
                    outputPtr.cast(outputPtr->cpuPtr)[writeIndex] = inputPtr.cast(inputPtr->cpuPtr)[i * kernel->groupSize];
                    writeIndex++;
                }
            }
            outputPtr->cpuLength = writeIndex;
        }, argPtrsTuple);
    };

    // Store the kernel information
    this->stages.emplace_back(str, cpuFunc, groupSize, inputLengthFraction, KernelArg(outputPtr.value, output, 0), argList);

    this->timingPoint("New stage (filter)");

    return true;
}

// Filter (InOut)
template<typename FilterType, typename... ArgTypes>
bool Pipeline::stage(const std::string &str, std::function<bool(FilterType*, ArgTypes*...)> func, ArgTyped<FilterType> inout, std::tuple<ArgTyped<ArgTypes>...> argsTuple, uint32_t overlap, uint32_t groupSize)
{
    DEBUG_HOST_V("Adding filter inout stage");
    this->timingPoint();

    // The first argument must be both, input and output
    if (!inout.isInOut())
    {
        this->hasFailed = true;
        ERROR_HOST("Filter argument is not InOut");
        return false;
    }

    if (groupSize > 1 || overlap > 0)
    {
        this->hasFailed = true;
        ERROR_HOST("Grouping/windowing filters may not have an InOut output");
        return false;
    }

    // Get the input length fraction in case the arguments won't occupy the whole space
    uint32_t inputLengthFraction = this->getInputLengthFraction(argsTuple);
    if (inputLengthFraction == -1)
        return false;

    // Do the operations common to all stage types
    auto argList = this->addArgs(argsTuple, overlap, groupSize, inputLengthFraction, inout);
    if (argList.empty())
        return false;

    // Check input flags
    if (hasInputsWithFlags(argList, ArgFlags::Grouped | ArgFlags::Reduced))
    {
        this->hasFailed = true;
        ERROR_HOST("Filter operation may not have grouped or reduced inputs");
        return false;
    }

    // The typed args tuple won't reflect changes made to the arguments from within the pipeline, so we need
    // to construct a tuple of pointers instead
    auto argPtrsTuple = std::apply(
        [this](auto &... arg){
            return std::make_tuple(TypeWrapper(arg, &this->args[arg.hostPtr])...);
        }, argsTuple);

    // Get typed pointers for the inout for the same reason as the tuple above
    auto inoutPtr = TypeWrapper(inout, &this->args[inout.hostPtr]);

    // Create a function that will process the CPU elements
    auto cpuFunc = [inoutPtr, func, argPtrsTuple](const Kernel *kernel){
        std::apply([inoutPtr, func, kernel](auto &&... arg) {
            // Copy the padding buffer for the current stage
            for (const auto &karg : kernel->args)
                if (karg.isInput() && karg.inputOverlap > 0)
                    std::memcpy(karg.base->cpuPtr + inoutPtr->cpuLength * karg.base->elementSize / kernel->inputLengthFraction,
                                karg.paddingBuffer.data(), karg.inputOverlap * karg.base->elementSize);

            uint64_t writeIndex = 0;
            for (uint64_t i = 0; i < inoutPtr->cpuLength; i++)
            {
                if (func((inoutPtr.cast(inoutPtr->cpuPtr) + i), (arg.cast(arg->cpuPtr) + (arg->isVector() ? i : 0))...))
                {
                    DEBUG_HOST_V("Writing filtered output: %lu -> %lu", i, writeIndex);
                    inoutPtr.cast(inoutPtr->cpuPtr)[writeIndex] = inoutPtr.cast(inoutPtr->cpuPtr)[i];
                    writeIndex++;
                }
            }
            inoutPtr->cpuLength = writeIndex;
        }, argPtrsTuple);
    };

    // Store the kernel information
    this->stages.emplace_back(str, cpuFunc, groupSize, inputLengthFraction, KernelArg(inoutPtr.value, inout, 0), argList);

    this->timingPoint("New stage (filter)");

    return true;
}

// Reduce
template<typename... ArgTypes>
bool Pipeline::stage(const std::string &str, std::function<void(ArgTypes*...)> func, std::tuple<ArgTyped<ArgTypes>...> argsTuple)
{
    DEBUG_HOST_V("Adding reduce stage");
    this->timingPoint();

    // If there are exactly 2 arguments, we can deduce the combine function if needed
    if constexpr (sizeof...(ArgTypes) == 2)
    {
        auto &first = std::get<0>(argsTuple);
        auto &second = std::get<1>(argsTuple);
        if constexpr (std::is_same<decltype(first), decltype(second)>::value)
        {
            // Use the kernel as a combine function
            auto combineFunc = [func, first, second](void *a, const void *b){
                using T = typename std::remove_reference<decltype(first)>::type::Type;
                func(reinterpret_cast<T*>(a), const_cast<T*>(reinterpret_cast<const T*>(b)));
            };
            if (first.isScalar() && first.isOutput() && second.isVector() && !second.isOutput() && !first.combineFunc)
                first.combineFunc = combineFunc;
            else if (first.isVector() && !first.isOutput() && second.isScalar() && second.isOutput() && !second.combineFunc)
                second.combineFunc = combineFunc;
        }
    }

    // Get the input length fraction in case the arguments won't occupy the whole space
    uint32_t inputLengthFraction = this->getInputLengthFraction(argsTuple);

    // Remove the types from the args and link them to the proper argument in the args list
    auto argList = this->addArgs(argsTuple, 0, 1, inputLengthFraction);
    if (argList.empty())
        return false;

    // No input parameter may have been a reduce output before
    if (std::any_of(argList.begin(), argList.end(),
                    [](const KernelArg &arg){ return arg.isInput() && arg.base->isReduced() && arg.isParameter(); }))
    {
        this->hasFailed = true;
        ERROR_HOST("Reduce operation may not have reduced inputs");
        return false;
    }

    // If any argument is a filtered vector, it must be the only vector argument
    uint64_t *lengthPtr = this->getReduceInputLengthPtrFromArgs(argList);
    if (lengthPtr == nullptr)
    {
        this->hasFailed = true;
        ERROR_HOST("Reductions may only have one or zero filtered inputs");
        return false;
    }

    // The typed args tuple won't reflect changes made to the arguments from within the pipeline, so we need
    // to construct a tuple of pointers instead
    auto argPtrsTuple = std::apply(
        [this](auto &... arg){
            return std::make_tuple(TypeWrapper(arg, &this->args[arg.hostPtr])...);
        }, argsTuple);

    // Create a function that will process an element on the CPU
    std::function cpuFunc = [func, argPtrsTuple, lengthPtr](const Kernel *kernel){
        std::apply([func, lengthPtr, kernel](auto &&... arg) {
            DEBUG_HOST("Reduce CPU length: %ld", *lengthPtr / kernel->inputLengthFraction);
            for (uint64_t i = 0; i < *lengthPtr / kernel->inputLengthFraction; i++)
                func((arg.cast(arg->cpuPtr) + (arg->isVector() ? i : 0))...);
        }, argPtrsTuple);
    };

    // Store the kernel information
    this->stages.emplace_back(str, cpuFunc, 1, inputLengthFraction, KernelArg(), argList);

    // Mark the kernel as a reduction
    this->stages.back().markReduce();

    this->timingPoint("New stage (reduce)");

    return true;
}

// Scan (inclusive). The combiner has the same (acc, x) signature as for Reduce. The args are
// an INOUT vector initialized with the identity element and an INPUT vector.
template<typename... ArgTypes>
bool Pipeline::stage(const std::string &str, std::function<void(ArgTypes*...)> func,
                     std::tuple<ArgTyped<ArgTypes>...> argsTuple, ScanTag)
{
    DEBUG_HOST_V("Adding scan stage");
    this->timingPoint();

    // Use the kernel as the combine function of the INOUT arg. The host needs it to
    // stitch the partial scans of different DPUs and rounds together.
    if constexpr (sizeof...(ArgTypes) == 2)
    {
        auto &first  = std::get<0>(argsTuple);
        auto &second = std::get<1>(argsTuple);
        if constexpr (std::is_same<decltype(first), decltype(second)>::value)
        {
            auto combineFunc = [func, first, second](void *a, const void *b){
                using T = typename std::remove_reference<decltype(first)>::type::Type;
                func(reinterpret_cast<T*>(a), const_cast<T*>(reinterpret_cast<const T*>(b)));
            };
            if (first.isVector() && first.isInput() && first.isOutput() && !first.combineFunc)
                first.combineFunc = combineFunc;
            else if (second.isVector() && second.isInput() && second.isOutput() && !second.combineFunc)
                second.combineFunc = combineFunc;
        }
    }

    // Get the input length fraction in case the arguments won't occupy the whole space
    uint32_t inputLengthFraction = this->getInputLengthFraction(argsTuple);

    // Remove the types from the args and link them to the proper argument in the args list
    auto argList = this->addArgs(argsTuple, 0, 1, inputLengthFraction);
    if (argList.empty())
        return false;

    // Mark the scan output
    for (auto &karg : argList)
    {
        if (karg.isVector() && karg.isInput() && karg.isOutput())
        {
            karg.base->markScanned();
            break;
        }
    }

    // The CPU part of a scan depends on the running prefix of the DPU part, so it is
    // computed in Pipeline::combineScans instead. This worker processes no elements.
    auto argPtrsTuple = std::apply(
        [this](auto &... arg){
            return std::make_tuple(TypeWrapper(arg, &this->args[arg.hostPtr])...);
        }, argsTuple);

    std::function cpuFunc = [func, argPtrsTuple](const Kernel *kernel){
        std::apply([func, kernel](auto &&... arg) {
            uint64_t length = kernel->inputLengthFraction > 0
                              ? (0)
                              : 0;
            for (uint64_t i = 0; i < length; i++)
                func((arg.cast(arg->cpuPtr) + (arg->isVector() ? i : 0))...);
        }, argPtrsTuple);
    };

    // Store the kernel and mark it as a scan
    this->stages.emplace_back(str, cpuFunc, 1, inputLengthFraction, KernelArg(), argList);
    this->stages.back().markScan();

    this->timingPoint("New stage (scan)");

    return true;
}

}
