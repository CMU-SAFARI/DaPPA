#include "catch.hpp"
#include "common.h"
#include "structs.h"

using namespace DaPPA;


TEST_CASE("1 byte", "[type]")
{
    std::string faultyStage = PIPELINE(64, // Max before overflow
                                       int32_t, int8_t, int8_t);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("8 bytes", "[type]")
{
    std::string faultyStage = PIPELINE(65536,
                                       int64_t, int64_t, int64_t);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("2/4/8 bytes", "[type]")
{
    std::string faultyStage = PIPELINE(16384, // Max before overflow
                                       int64_t, int16_t, int32_t);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("4 byte struct", "[type]")
{
    std::string faultyStage = PIPELINE_STRUCT(128, // Max before overflow
                                              struct Test4Bytes, struct Test4Bytes, struct Test4Bytes);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("16 byte struct", "[type]")
{
    std::string faultyStage = PIPELINE_STRUCT(65536,
                                              struct Test16Bytes, struct Test16Bytes, struct Test16Bytes);

    REQUIRE(faultyStage == "none");
}

TEST_CASE("10 byte struct (auto-pad)", "[type]")
{
    std::string faultyStage = PIPELINE_STRUCT(256, // Max before overflow
                                              struct Test10Bytes, struct Test10Bytes, struct Test10Bytes);

    REQUIRE(faultyStage == "none");
}
