#ifndef DAPPA_STRUCTS_H
#define DAPPA_STRUCTS_H

#ifdef __cplusplus
    #include <ostream>

    #define TEST_STRUCT_OPERATORS(T, SmallestT) \
        using SmallestIntT = SmallestT; \
        bool operator ==(const SmallestIntT &other) const { return this->a == other && this->b == other; } \
        T operator +() const { return *this; } \
        T &operator =(const SmallestIntT &val)  { this->a = val; this->b = val; return *this; } \
        void print(std::ostream &os) const { os << this->a << ' ' << this->b; }

    // Make all classes with a 'print' function printable via an ostream
    template<class T>
    auto operator<<(std::ostream &os, const T &t) -> decltype(t.print(os), os)
    {
        t.print(os);
        return os;
    }
#else
    #define TEST_STRUCT_OPERATORS(T, SmallestIntT)
#endif

struct Test4Bytes
{
    int16_t a;
    int16_t b;

    TEST_STRUCT_OPERATORS(Test4Bytes, int16_t)
};

struct Test16Bytes
{
    int64_t a;
    int64_t b;

    TEST_STRUCT_OPERATORS(Test16Bytes, int64_t)
};

struct Test10Bytes
{
    int64_t a;
    int16_t b;

    TEST_STRUCT_OPERATORS(Test10Bytes, int16_t)
};

struct TestReduceResult
{
    int64_t sum;
    int32_t length;
};

// Linear-regression point: 10 features + 1 label (same layout as SimplePIM).
struct LinRegPoint
{
    int32_t feat[10];
    int32_t label;
};

// Linear-regression gradient accumulator.
struct LinRegGrad
{
    int64_t g[10];
};

// K-Means point: 10 coordinates.
struct KMeansPoint
{
    int32_t feat[10];
};

// K-Means per-centroid bucket: point count + coordinate sums.
struct KMeansBucket
{
    int32_t count;
    int32_t sum[10];
};

// K-Means reduction accumulator: one bucket per centroid (k = 10).
struct KMeansResult
{
    struct KMeansBucket b[10];
};

// Logistic-regression point: 10 features + 1 label.
struct LogRegPoint
{
    float feat[10];
    float label;
};

// Logistic-regression gradient accumulator.
struct LogRegGrad
{
    float g[10];
};

#endif //DAPPA_STRUCTS_H
