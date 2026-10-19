#include "catch.hpp"
#include "common.h"

using namespace DaPPA;

#define CHECK_COMBINATION(target_length) checkVector4x(file, out, p.getLength(out), target_length)

const static int32_t length = 32769;

TEST_CASE("Map -> Map", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, &ans),
                INPUT(uint32_t, in)));

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, out),
                INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Map -> Filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    int32_t ans;
    std::vector<int32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(MAP(([](int32_t *out, int32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(int32_t, &ans),
                INPUT(int32_t, in)));

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, out),
                   INPUT(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION((length - 1) / 2 + 1));
}

TEST_CASE("Map -> Reduce", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    uint32_t out = 0;

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, &ans),
                INPUT(uint32_t, in)));

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &out),
                   INPUT(uint32_t, &ans)));

    p.fetch(&out);
    p.execute();

    REQUIRE(out == arithSum(length, 2 * (length - 1)));
}

TEST_CASE("Map -> Window", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);
    std::vector<uint32_t> padding = {length * 2, (length + 1) * 2, (length + 2) * 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, &ans),
                INPUT(uint32_t, in)));

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                        *out = in[0] + in[3] - 6;
                    }),
                    4,
                    OUTPUT(uint32_t, out),
                    INPUT(uint32_t, &ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Map -> Group", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, &ans),
                INPUT(uint32_t, in)));

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[3] - 6) / 4;
                  }),
                  4,
                  OUTPUT(uint32_t, out),
                  INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Map -> Window filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    std::vector<int32_t> out(length);
    int32_t ans;
    std::vector<int32_t> padding = {length / 2, length / 2 + 2, length / 2 + 4};

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(MAP(([](int32_t *out, int32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(int32_t, &ans),
                INPUT(int32_t, in)));

    p.stage(WINDOW_FILTER(([](int32_t *in) {
                              return in[0] >= 0 && in[3] <= 0;
                          }),
                          4,
                          OUTPUT(int32_t, out),
                          INPUT(int32_t, &ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 2 - 1));
}

TEST_CASE("Map -> Group filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    std::vector<int32_t> out(length);
    int32_t ans;

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(MAP(([](int32_t *out, int32_t *in) {
                    *out = *in >= 0 ? *in : -*in;
                }),
                OUTPUT(int32_t, &ans),
                INPUT(int32_t, in)));

    p.stage(GROUP_FILTER(([](int32_t *in) {
                             return *in % 4 == 0;
                         }),
                         2,
                         OUTPUT(int32_t, out),
                         INPUT(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Map -> Window group", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    std::vector<uint32_t> out(length);
    uint32_t ans;
    std::vector<uint32_t> padding = {length * 2, length * 2 + 2, length * 2 + 4};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, &ans),
                INPUT(uint32_t, in)));

    p.stage(WINDOW_GROUP(([](uint32_t *out, uint32_t *in) {
                             *out = (in[0] + in[3] - 6) / 2;
                         }),
                         2,
                         4,
                         OUTPUT(uint32_t, out),
                         INPUT(uint32_t, &ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 2));
}

TEST_CASE("Map -> Window group filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    std::vector<int32_t> out(length);
    int32_t ans;
    std::vector<int32_t> padding = {length * 2, length * 2 + 2, length * 2 + 4};

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(MAP(([](int32_t *out, int32_t *in) {
                    *out = *in;
                }),
                OUTPUT(int32_t, &ans),
                INPUT(int32_t, in)));

    p.stage(WINDOW_GROUP_FILTER(([](int32_t *in) {
                                    return in[0] >= 0 && in[3] <= 0 && in[0] % 4 == 0;
                                }),
                                2,
                                4,
                                OUTPUT(int32_t, out),
                                INPUT(int32_t, &ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Filter -> Map", "[combinations]")
{
    PipelineFull p(length);

    std::vector<int32_t> filter_in(length);
    std::vector<int32_t> map_in(length);
    std::vector<int32_t> out(length);

    for (int32_t i = 0; i < length; i++)
    {
        filter_in[i] = negativeOdds(i);
        map_in[i] = 2 * i;
    }

    p.stage(FILTER(([](int32_t *data) {
                    return *data >= 0;
                }),
                INOUT(int32_t, filter_in)));

    p.stage(MAP(([](int32_t *out, int32_t *a, int32_t *b) {
                    *out = *a + *b;
                }),
                OUTPUT(int32_t, out),
                INPUT(int32_t, filter_in),
                INPUT(int32_t, map_in)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 2 + 1));
}

TEST_CASE("Filter -> Filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    int32_t ans;
    std::vector<int32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, &ans),
                   INPUT(int32_t, in)));

    p.stage(FILTER(([](int32_t *in) {
                       return *in % 4 == 0;
                   }),
                   OUTPUT(int32_t, out),
                   INPUT(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION((length - 1) / 4 + 1));
}

TEST_CASE("Filter -> Reduce", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    int32_t ans;
    int32_t out = 0;

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, &ans),
                   INPUT(int32_t, in)));

    p.stage(REDUCE(([](int32_t *out, int32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(int32_t, &out),
                   INPUT(int32_t, &ans)));

    p.fetch(&out);
    p.execute();

    REQUIRE(out == arithSum((length + 1) / 2, ((length - 1) / 2) * 2));
}

TEST_CASE("Filter -> Window", "[combinations]")
{
    PipelineFull p(length);

    std::vector<int32_t> in(length);
    std::vector<int32_t> ans(length);
    std::vector<int32_t> out(length);
    std::vector<int32_t> padding = {length + 1, length + 3, length + 5};

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, ans),
                   INPUT(int32_t, in)));
    p.stage(WINDOW(([](int32_t *out, int32_t *in) {
                        *out = in[0] + in[3] - 6;
                    }),
                    4,
                    OUTPUT(int32_t, out),
                    INPUT(int32_t, ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 2 + 1));
}

TEST_CASE("Filter -> Group", "[combinations]")
{
    PipelineFull p(length);

    std::vector<int32_t> in(length);
    std::vector<int32_t> ans(length);
    std::vector<int32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, ans),
                   INPUT(int32_t, in)));

    p.stage(GROUP(([](int32_t *out, int32_t *in) {
                      *out = (in[0] + in[3] - 6) / 4;
                  }),
                  4,
                  OUTPUT(int32_t, out),
                  INPUT(int32_t, ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 8));
}

TEST_CASE("Reduce -> Map", "[combinations]")
{
    PipelineFull p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans = 0;
    std::vector<uint32_t> out(length);

    uint32_t sum = arithSum(length, length - 1);
    for (uint32_t i = 0; i < length; i++)
    {
        in[i] = i;
        out[i] = 4 * i + sum;
    }

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &ans),
                   INPUT(uint32_t, in)));

    p.stage(MAP(([](uint32_t *out, uint32_t *param) {
                    *out -= *param;
                }),
                INOUT(uint32_t, out),
                PARAMETER(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Reduce -> Filter", "[combinations]")
{
    PipelineFull p(length);

    std::vector<int32_t> in_map(length);
    std::vector<int32_t> in_filter(length);
    int32_t ans = 0;
    std::vector<int32_t> out(length);

    uint32_t sum = arithSum(length, length - 1);
    for (uint32_t i = 0; i < length; i++)
    {
        in_map[i] = i;
        in_filter[i] = 2 * negativeOdds(i) + sum;
    }

    p.stage(REDUCE(([](int32_t *out, int32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(int32_t, &ans),
                   INPUT(int32_t, in_map)));

    p.stage(FILTER(([](int32_t *in, int32_t *param) {
                       *in -= *param;
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, out),
                   INPUT(int32_t, in_filter),
                   PARAMETER(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 2 + 1));
}

TEST_CASE("Reduce -> Reduce", "[combinations]")
{
    PipelineFull p(length);

    std::vector<uint64_t> in(length);
    uint64_t ans = 0;
    uint64_t out = 0;

    for (uint64_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(REDUCE(([](uint64_t *out, uint64_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint64_t, &ans),
                   INPUT(uint64_t, in)));

    p.stage(REDUCE(([](uint64_t *out, uint64_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint64_t, &out,
                              ([](uint64_t *out, const uint64_t *in) {
                                  *out += *in;
                              })),
                   PARAMETER(uint64_t, &ans)));

    p.fetch(&out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(out == length * arithSum(length, length - 1));
}

TEST_CASE("Reduce -> Window", "[combinations]")
{
    PipelineFull p(length);

    std::vector<uint32_t> in_map(length);
    std::vector<uint32_t> in_window(length);
    uint32_t ans = 0;
    std::vector<uint32_t> out(length);
    const uint32_t sum = arithSum(length, length - 1);
    std::vector<uint32_t> padding = {length + sum, length + 1 + sum, length + 2 + sum};

    for (uint32_t i = 0; i < length; i++)
    {
        in_map[i] = i;
        in_window[i] = i + sum;
    }

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &ans),
                   INPUT(uint32_t, in_map)));

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in, uint32_t *param) {
                        *out = 4 * (in[0] + in[3] - in[2] - 1 - *param);
                    }),
                    4,
                    OUTPUT(uint32_t, out),
                    INPUT(uint32_t, in_window, padding),
                    PARAMETER(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Reduce -> Group", "[combinations]")
{
    PipelineFull p(length);

    std::vector<uint32_t> in_map(length);
    std::vector<uint32_t> in_group(length);
    uint32_t ans = 0;
    std::vector<uint32_t> out(length);

    const uint32_t sum = arithSum(length, length - 1);
    for (uint32_t i = 0; i < length; i++)
    {
        in_map[i] = i;
        in_group[i] = i + sum;
    }

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &ans),
                   INPUT(uint32_t, in_map)));

    p.stage(GROUP(([](uint32_t *out, uint32_t *in, uint32_t *param) {
                      *out = in[0] + in[3] - in[2] - 1 - *param;
                  }),
                  4,
                  OUTPUT(uint32_t, out),
                  INPUT(uint32_t, in_group),
                  PARAMETER(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Window -> Map", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);
    std::vector<uint32_t> padding = {length, length + 1, length + 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                       *out = in[0] + in[3] - 3;
                   }),
                   4,
                   OUTPUT(uint32_t, &ans),
                   INPUT(uint32_t, in, padding)));

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, out),
                INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Window -> Filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    int32_t ans;
    std::vector<int32_t> out(length);
    std::vector<int32_t> padding = {length, length + 1, length + 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = negativeOdds(i);

    p.stage(WINDOW(([](int32_t *out, int32_t *in) {
                       *out = in[0] + in[2] - 2;
                   }),
                   4,
                   OUTPUT(int32_t, &ans),
                   INPUT(int32_t, in, padding)));

    p.stage(FILTER(([](int32_t *in) {
                       return *in >= 0;
                   }),
                   OUTPUT(int32_t, out),
                   INPUT(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION((length - 1) / 2) + 1);
}

TEST_CASE("Window -> Reduce", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    uint32_t out = 0;
    std::vector<uint32_t> padding = {length, length + 1, length + 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                       *out = in[0] + in[3] - 3;
                   }),
                   4,
                   OUTPUT(uint32_t, &ans),
                   INPUT(uint32_t, in, padding)));

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &out),
                   INPUT(uint32_t, &ans)));

    p.fetch(&out);
    p.execute();

    REQUIRE(out == arithSum(length, (length - 1) * 2));
}

TEST_CASE("Window -> Window", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);
    std::vector<uint32_t> padding1 = {length, length + 1, length + 2};
    std::vector<uint32_t> padding2 = {length * 2, (length + 1) * 2, (length + 2) * 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                       *out = in[0] + in[3] - 3;
                   }),
                   4,
                   OUTPUT(uint32_t, &ans),
                   INPUT(uint32_t, in, padding1)));

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                        *out = in[0] + in[3] - 6;
                    }),
                    4,
                    OUTPUT(uint32_t, out),
                    INPUT(uint32_t, &ans, padding2)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length));
}

TEST_CASE("Window -> Group", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);
    std::vector<uint32_t> padding = {length, length + 1, length + 2};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                       *out = in[0] + in[3] - 3;
                   }),
                   4,
                   OUTPUT(uint32_t, &ans),
                   INPUT(uint32_t, in, padding)));

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[3] - 6) / 4;
                  }),
                  4,
                  OUTPUT(uint32_t, out),
                  INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Group -> Map", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[3] - 3) / 4;
                  }),
                  4,
                  OUTPUT(uint32_t, &ans),
                  INPUT(uint32_t, in)));

    p.stage(MAP(([](uint32_t *out, uint32_t *in) {
                    *out = *in * 2;
                }),
                OUTPUT(uint32_t, out),
                INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Group -> Filter", "[combinations]")
{
    Pipeline p(length);

    std::vector<int32_t> in(length);
    int32_t ans;
    std::vector<int32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(GROUP(([](int32_t *out, int32_t *in) {
                      *out = (in[0] + in[1] - 1) / 2;
                  }),
                  2,
                  OUTPUT(int32_t, &ans),
                  INPUT(int32_t, in)));

    p.stage(FILTER(([](int32_t *in) {
                       return *in % 4 == 0;
                   }),
                   OUTPUT(int32_t, out),
                   INPUT(int32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Group -> Reduce", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    uint32_t out = 0;

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[3] - 3) / 4;
                  }),
                  4,
                  OUTPUT(uint32_t, &ans),
                  INPUT(uint32_t, in)));

    p.stage(REDUCE(([](uint32_t *out, uint32_t *in) {
                       *out += *in;
                   }),
                   REDUCE_OUT(uint32_t, &out),
                   INPUT(uint32_t, &ans)));

    p.fetch(&out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(out == arithSum(length / 4, ((length / 4) - 1) * 2));
}

TEST_CASE("Group -> Window", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);
    std::vector<uint32_t> padding = {length / 2, length / 2 + 2, length / 2 + 4};

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[3] - 3) / 4;
                  }),
                  4,
                  OUTPUT(uint32_t, &ans),
                  INPUT(uint32_t, in)));

    p.stage(WINDOW(([](uint32_t *out, uint32_t *in) {
                        *out = in[0] + in[3] - 6;
                    }),
                    4,
                    OUTPUT(uint32_t, out),
                    INPUT(uint32_t, &ans, padding)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}

TEST_CASE("Group -> Group", "[combinations]")
{
    Pipeline p(length);

    std::vector<uint32_t> in(length);
    uint32_t ans;
    std::vector<uint32_t> out(length);

    for (uint32_t i = 0; i < length; i++)
        in[i] = i;

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[1] - 1) / 2;
                  }),
                  2,
                  OUTPUT(uint32_t, &ans),
                  INPUT(uint32_t, in)));

    p.stage(GROUP(([](uint32_t *out, uint32_t *in) {
                      *out = (in[0] + in[1] - 1) / 2;
                  }),
                  2,
                  OUTPUT(uint32_t, out),
                  INPUT(uint32_t, &ans)));

    p.fetch(out);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_COMBINATION(length / 4));
}
