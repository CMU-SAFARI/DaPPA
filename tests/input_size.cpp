#include "catch.hpp"
#include "common.h"

using namespace DaPPA;


TEST_CASE("Less elements than tasklets", "[size]")
{
    std::string faultyStage = PIPELINE(Device::dpuCountAvailable + 1,
                                       int32_t, int32_t, int32_t);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("Single round, CPU tasks", "[size]")
{
    std::string faultyStage = PIPELINE(Device::dpuCountAvailable * Device::taskletCountPerDpu * 10 + 3,
                                       int32_t, int32_t, int32_t);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("Multiple rounds", "[size]")
{
    std::string faultyStage = PIPELINE(Device::dpuCountAvailable * Device::mramSize / 15,
                                       int32_t, int32_t, int32_t);

    REQUIRE(faultyStage == "none");
}
