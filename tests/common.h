#include <vector>
#include <fstream>
#include "dappa.h"
#include "structs.h"
#include "logging.h"

#if VERBOSITY >= VERBOSITY_DEBUG
    #define OPEN_TEST_RESULT_FILE() std::ofstream file("result.txt")
    #define CHECK_TEST_RESULT(actual, target, tag) checkTestResult(file, tag, actual, target)

    template<typename T1, typename T2>
    bool checkTestResult(std::ofstream &file, const std::string &tag, const T1 &actual, const T2 &target)
    {
        bool correct = ((actual) == (target));
        // Using unary '+' to print 1 byte types as numbers
        file << tag << ": " << +actual;
        if (!correct)
            file << " (Wrong, should be " << +target << ")";
        file << std::endl;
        return correct;
    }

    bool printTestHeader(std::ofstream &file, const std::string &stage);
    bool printTestHeader(std::ofstream &file, const std::string &stage, uint64_t actual, uint64_t target);

    #define PRINT_TEST_HEADER(stage, ...) printTestHeader(file, stage __VA_OPT__(,) __VA_ARGS__)
#else
    #define OPEN_TEST_RESULT_FILE() std::ofstream file
    #define CHECK_TEST_RESULT(actual, target, tag) (actual == target)
    #define PRINT_TEST_HEADER(stage, ...) (true)
#endif

template<typename IntT>
bool checkVector4x(std::ofstream &file, std::vector<IntT> vector, uint64_t actual_length, uint64_t target_length)
{
    if (!PRINT_TEST_HEADER("result", actual_length, target_length))
        return false;

    bool stageCorrect = true;
    for (uint64_t i = 0; i < target_length; i++)
        stageCorrect = CHECK_TEST_RESULT(vector[i], i * 4, std::to_string(i)) && stageCorrect;

    return stageCorrect;
}

#define CHECK_VECTOR(stage, actual_length, target_length, actual_value, target_value) \
do { \
    bool stageCorrect = true; \
    if (!PRINT_TEST_HEADER(stage, actual_length, target_length)) \
        return (stage); \
    for (int64_t i = 0; i < (target_length); i++) \
        stageCorrect = CHECK_TEST_RESULT(actual_value, target_value, std::to_string(i)) && stageCorrect; \
    if (!stageCorrect) \
        return (stage); \
} while(0)

#define CHECK_REDUCE(stage, actual, target) \
do { \
    PRINT_TEST_HEADER(stage); \
    uint32_t i = 0; \
    if (!CHECK_TEST_RESULT(actual, target, #actual)) \
        return (stage); \
} while(0)

#define CHECK_REDUCE_STRUCT(stage, actual1, target1, actual2, target2) \
do { \
    PRINT_TEST_HEADER(stage); \
    int64_t i = 0; \
    if (!CHECK_TEST_RESULT(actual1, target1, #actual1)) \
        return (stage); \
    i++; \
    if (!CHECK_TEST_RESULT(actual2, target2, #actual2)) \
        return (stage); \
} while(0)

constexpr int64_t negativeOdds(uint64_t x)
{
    return x * (1 + (-2 * (x % 2)));
}

constexpr uint64_t arithSum(uint64_t length, uint64_t largest)
{
    return (length * largest) / 2;
}

#define PIPELINE(in_length, TR, T1, T2) \
[]() -> std::string \
{ \
    uint64_t length = DaPPA::Device::paddedSizeWram(in_length); \
    Pipeline p(length); \
    p.addHeader(testsStructsHeaderBinData); \
    std::vector<T1> padding = {static_cast<T1>(length), static_cast<T1>(length + 1)}; \
\
    std::vector<TR> mapOut(length); \
    std::vector<T1> mapInA(length); \
    std::vector<T2> mapInB(length); \
\
    std::vector<TR> filterOut(length); \
    std::vector<TR> &filterIn = mapOut; \
    std::vector<TR> filterInOut(length); \
\
    std::vector<T1> windowIn(length); \
    std::vector<TR> windowOut(length); \
\
    std::vector<T1> groupIn(length); \
    std::vector<TR> groupOut(length); \
\
    TR simpleReduction{}; \
    struct TestReduceResult complexReduction{}; \
\
    for (uint64_t i = 0; i < length; i++) \
    { \
        mapInA[i] = negativeOdds(i); \
        mapInB[i] = negativeOdds(i); \
        filterInOut[i] = negativeOdds(i); \
        windowIn[i] = i; \
        groupIn[i] = i; \
        mapOut[i] = 0; \
        filterOut[i] = 0; \
        windowOut[i] = 0; \
        groupOut[i] = 0; \
    } \
\
    p.stage(MAP(([](TR *c, T1 *a, T2 *b){ \
                    *c = *a + *b; \
                }), \
                OUTPUT(TR, mapOut), \
                INPUT(T1, mapInA), \
                INPUT(T2, mapInB))); \
\
    p.stage(FILTER(([](TR *in){ \
                       return *in >= 0; \
                   }), \
                   OUTPUT(TR, filterOut), \
                   INPUT(TR, filterIn))); \
\
    p.stage(FILTER(([](TR *in){ \
                       return *in >= 0; \
                   }), \
                   INOUT(TR, filterInOut))); \
\
    p.stage(REDUCE(([](TR *out, TR *in){ \
                *out += *in; \
            }), \
            REDUCE_OUT(TR, &simpleReduction), \
            INPUT(TR, filterOut))); \
\
    p.stage(REDUCE(([](struct TestReduceResult *out, TR *in){ \
                out->length++; \
                out->sum += *in; \
            }), \
            REDUCE_OUT(struct TestReduceResult, &complexReduction, \
                       ([](struct TestReduceResult *out, const struct TestReduceResult *in) { \
                           out->length += in->length; \
                           out->sum += in->sum; \
                       })), \
            INPUT(TR, filterInOut))); \
\
    p.stage(WINDOW([](TR *out, T1 *in){ \
                       *out = in[0] + in[1] + in[2]; \
                   }, \
                   3, \
                   OUTPUT(TR, windowOut), \
                   INPUT(T1, windowIn, padding))); \
\
    p.stage(GROUP([](TR *out, T1 *in){ \
                       *out = in[0] + in[1] + in[2] + in[3]; \
                   }, \
                   4, \
                   OUTPUT(TR, groupOut), \
                   INPUT(T1, groupIn))); \
\
    p.fetch(mapOut); \
    p.fetch(filterOut); \
    p.fetch(filterInOut); \
    p.fetch(&simpleReduction); \
    p.fetch(&complexReduction); \
    p.fetch(windowOut); \
    p.fetch(groupOut); \
    p.execute(); \
\
    OPEN_TEST_RESULT_FILE(); \
    CHECK_VECTOR("map", p.getLength(mapOut), length, mapOut[i], 2 * negativeOdds(i)); \
    CHECK_VECTOR("filter input/output", p.getLength(filterOut), (length + 1) / 2, filterOut[i], 4 * i); \
    CHECK_VECTOR("filter inout", p.getLength(filterInOut), (length + 1) / 2, filterInOut[i], 2 * i); \
    CHECK_REDUCE("simple reduce", simpleReduction, static_cast<TR>(arithSum((length + 1) / 2, ((length - 1) / 2) * 4))); \
    CHECK_REDUCE_STRUCT("complex reduce", \
                        complexReduction.length, (length + 1) / 2, \
                        complexReduction.sum, arithSum((length + 1) / 2, ((length - 1) / 2) * 2)); \
    CHECK_VECTOR("window", p.getLength(windowOut), length, windowOut[i], 3 * (i + 1)); \
    CHECK_VECTOR("group", p.getLength(groupOut), length / 4, groupOut[i], 6 + 16 * i); \
\
    return "none"; \
}()

#define PIPELINE_STRUCT(in_length, TR, T1, T2) \
[]() -> std::string \
{ \
    uint64_t length = DaPPA::Device::paddedSizeWram(in_length); \
    Pipeline p(length); \
    p.addHeader(testsStructsHeaderBinData); \
\
    std::vector<TR> mapOut(length); \
    std::vector<T1> mapInA(length); \
    std::vector<T2> mapInB(length); \
\
    std::vector<TR> filterOut(length); \
    std::vector<TR> &filterIn = mapOut; \
    std::vector<TR> filterInOut(length); \
\
    std::vector<T1> windowIn(length); \
    std::vector<TR> windowOut(length); \
\
    std::vector<T1> groupIn(length); \
    std::vector<TR> groupOut(length); \
\
    using SmallestIntT = typename decltype(windowIn)::value_type::SmallestIntT; \
    std::vector<T1> padding = { {static_cast<SmallestIntT>(length), static_cast<SmallestIntT>(length)}, \
                                {static_cast<SmallestIntT>(length + 1), static_cast<SmallestIntT>(length + 1)} }; \
\
    TR simpleReduction{}; \
    struct TestReduceResult complexReduction{}; \
\
    for (uint64_t i = 0; i < length; i++) \
    { \
        mapInA[i] = negativeOdds(i); \
        mapInB[i] = negativeOdds(i); \
        filterInOut[i] = negativeOdds(i); \
        windowIn[i] = i; \
        groupIn[i] = i; \
        mapOut[i] = 0; \
        filterOut[i] = 0; \
        windowOut[i] = 0; \
        groupOut[i] = 0; \
    } \
\
    p.stage(MAP([](TR *c, T1 *a, T2 *b){ \
                    c->a = a->a + b->a; \
                    c->b = a->b + b->b; \
                }, \
                OUTPUT(TR, mapOut), \
                INPUT(T1, mapInA), \
                INPUT(T2, mapInB))); \
\
    p.stage(FILTER([](TR *in){ \
                       return in->a >= 0 && in->b >= 0; \
                   }, \
                   OUTPUT(TR, filterOut), \
                   INPUT(TR, filterIn))); \
\
    p.stage(FILTER([](TR *in){ \
                       return in->a >= 0 && in->b >= 0; \
                   }, \
                   INOUT(TR, filterInOut))); \
\
    p.stage(REDUCE(([](TR *out, TR *in){ \
                       out->a += in->a; \
                       out->b += in->b; \
                   }), \
            REDUCE_OUT(TR, &simpleReduction), \
            INPUT(TR, filterOut))); \
\
    p.stage(REDUCE(([](struct TestReduceResult *out, TR *in){ \
                       out->length++; \
                       out->sum += in->a + in->b; \
                   }), \
            REDUCE_OUT(struct TestReduceResult, &complexReduction, \
                       ([](struct TestReduceResult *out, const struct TestReduceResult *in) { \
                           out->length += in->length; \
                           out->sum += in->sum; \
                       })), \
            INPUT(TR, filterInOut))); \
\
    p.stage(WINDOW([](TR *out, T1 *in){ \
                       out->a = in[0].a + in[1].a + in[2].a; \
                       out->b = in[0].b + in[1].b + in[2].b; \
                   }, \
                   3, \
                   OUTPUT(TR, windowOut), \
                   INPUT(T1, windowIn, padding))); \
\
    p.stage(GROUP([](TR *out, T1 *in){ \
                       out->a = in[0].a + in[1].a + in[2].a + in[3].a; \
                       out->b = in[0].b + in[1].b + in[2].b + in[3].b; \
                   }, \
                   4, \
                   OUTPUT(TR, groupOut), \
                   INPUT(T1, groupIn))); \
\
    p.fetch(mapOut); \
    p.fetch(filterOut); \
    p.fetch(filterInOut); \
    p.fetch(&simpleReduction); \
    p.fetch(&complexReduction); \
    p.fetch(windowOut); \
    p.fetch(groupOut); \
    p.execute(); \
\
    OPEN_TEST_RESULT_FILE(); \
    CHECK_VECTOR("map", p.getLength(mapOut), length, mapOut[i], 2 * negativeOdds(i)); \
    CHECK_VECTOR("filter input/output", p.getLength(filterOut), (length + 1) / 2, filterOut[i], 4 * i); \
    CHECK_VECTOR("filter inout", p.getLength(filterInOut), (length + 1) / 2, filterInOut[i], 2 * i); \
    CHECK_REDUCE_STRUCT("simple reduce", \
                        simpleReduction.a, arithSum((length + 1) / 2, ((length - 1) / 2) * 4), \
                        simpleReduction.b, arithSum((length + 1) / 2, ((length - 1) / 2) * 4)); \
    CHECK_REDUCE_STRUCT("complex reduce", \
                        complexReduction.length, (length + 1) / 2, \
                        complexReduction.sum, arithSum((length + 1) / 2, ((length - 1) / 2) * 4)); \
    CHECK_VECTOR("window", p.getLength(windowOut), length, windowOut[i], 3 * (i + 1)); \
    CHECK_VECTOR("group", p.getLength(groupOut), length / 4, groupOut[i], 6 + 16 * i); \
\
    return "none"; \
}()

extern "C"
{
    #define INCBIN_PREFIX
    #include "incbin.h"
    // Embedded copy of structs.h, passed to the generated DPU code
    INCTXT_EXTERN(testsStructsHeaderBin);
}
