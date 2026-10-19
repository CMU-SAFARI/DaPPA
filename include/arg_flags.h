#ifndef DAPPA_ARG_FLAGS_H
#define DAPPA_ARG_FLAGS_H

#include <cstdint>

namespace DaPPA
{

enum class ArgFlags : uint32_t
{
    Invalid     = 0,
    Input       = 1 << 1,
    Output      = 1 << 2,
    InOut       = 1 << 3,
    Vector      = 1 << 4,
    Scalar      = 1 << 5,
    Filtered    = 1 << 6,
    Fetched     = 1 << 7,
    Pushed      = 1 << 8,
    Windowed    = 1 << 9,
    Grouped     = 1 << 10,
    Reduced     = 1 << 11,
    Parameter   = 1 << 12,
    OutOfBounds = 1 << 13,
    Scanned     = 1 << 14,
    // Lookup table: a randomly indexed input that is broadcast to the MRAM of every DPU
    // and read with LOOKUP(). It is neither sharded (Vector) nor copied to WRAM (Scalar).
    Lookup      = 1 << 15
};

constexpr enum ArgFlags operator |(const enum ArgFlags &first, const enum ArgFlags &second)
{
    return static_cast<enum ArgFlags>(static_cast<uint32_t>(first) | static_cast<uint32_t>(second));
}

constexpr bool operator &(const enum ArgFlags &first, const enum ArgFlags &second)
{
    return static_cast<uint32_t>(first) & static_cast<uint32_t>(second);
}

class ArgFlagsBase
{
public:
    explicit ArgFlagsBase(ArgFlags flags);

    [[nodiscard]] bool isValid() const;
    [[nodiscard]] bool isInput() const;
    [[nodiscard]] bool isOutput() const;
    [[nodiscard]] bool isInOut() const;
    [[nodiscard]] bool isVector() const;
    [[nodiscard]] bool isScalar() const;
    [[nodiscard]] bool isFiltered() const;
    [[nodiscard]] bool isFetched() const;
    [[nodiscard]] bool isPushed() const;
    [[nodiscard]] bool isWindowed() const;
    [[nodiscard]] bool isReduced() const;
    [[nodiscard]] bool isParameter() const;
    [[nodiscard]] bool isScanned() const;
    [[nodiscard]] bool isLookup() const;

    void markFiltered();
    void markFetched();
    void markPushed();
    void markWindowed();
    void markGrouped();
    void markScanned();

    ArgFlags flags;
};

}

#endif //DAPPA_ARG_FLAGS_H
