#include "catch.hpp"
#include "common.h"
#include <algorithm>

using namespace DaPPA;

static constexpr uint32_t lengthScale = 1;
static std::mt19937 rngAlgorithm;

#define CHECK_RESULT(stage, actual_length, target_length, actual_value, target_value) \
        [&]() -> std::string {CHECK_VECTOR(stage, actual_length, target_length, actual_value, target_value); return "none";}()


template<typename T>
void randomize(std::vector<T> &vector, int32_t max = 1024)
{
    std::uniform_int_distribution<> rng(0, max);
    for (auto &element : vector)
        element = rng(rngAlgorithm);
}

template<typename T>
void clear(std::vector<T> &vector)
{
    for (auto &element : vector)
        element = 0;
}


TEST_CASE("VA: Vector addition", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<int32_t> a(length);
    std::vector<int32_t> b(length);
    std::vector<int32_t> c(length);

    randomize(a);
    randomize(b);
    clear(c);

    Pipeline p(length);

    p.stage(MAP([](int32_t *a, int32_t *b, int32_t *c){
                    *c = *a + *b;
                },
                INPUT(int32_t, a),
                INPUT(int32_t, b),
                OUTPUT(int32_t, c)));

    p.fetch(c);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("group", p.getLength(c), length, c[i], a[i] + b[i]) == "none");
}

TEST_CASE("GEMV: Vector-Matrix Multiplication", "[workload]")
{
    uint32_t rows = 4096 * lengthScale;
    uint32_t cols = 64;   // Same matrix width as the PrIM GEMV baseline
    std::vector<uint32_t> M(rows * cols);
    std::vector<uint32_t> a(cols);
    std::vector<uint32_t> b(rows);

    randomize(M);
    randomize(a);
    clear(b);

    Pipeline p(rows * cols);

    p.stage(GROUP([](uint32_t *M, uint32_t *a, uint32_t *b, uint32_t *cols){
                      uint32_t res = 0;
                      for (uint32_t i = 0; i < *cols; i++)
                          res += M[i] * a[i];
                      *b = res;
                  },
                  cols,
                  INPUT(uint32_t, M),
                  PARAMETER(uint32_t, a, sizeof(uint32_t) * cols),
                  OUTPUT(uint32_t, b),
                  PARAMETER(uint32_t, &cols)));

    p.fetch(b);
    p.execute();

    std::vector<uint32_t> cpuResult(rows);
    for (uint32_t i = 0; i < rows; i++)
    {
        uint32_t sum = 0;
        for (uint32_t j = 0; j < cols; j++)
            sum += M[i * cols + j] * a[j];
        cpuResult[i] = sum;
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("group", p.getLength(b), rows, b[i], cpuResult[i]) == "none");
}

TEST_CASE("SEL: Selection", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<uint64_t> x(length);

    randomize(x);

    std::vector<uint64_t> cpuResult;
    for (auto element : x)
        if (element % 2 == 0)
            cpuResult.push_back(element);

    Pipeline p(length);

    p.stage(FILTER([](uint64_t *x){
                       return *x % 2 == 0;
                   },
                   INOUT(uint64_t, x)));

    p.fetch(x);
    p.execute();

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("filter", p.getLength(x), p.getLength(x), x[i], cpuResult[i]) == "none");
}

TEST_CASE("UNI: Unique", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<int64_t> in(length);
    std::vector<int64_t> out(length);

    for (int64_t i = 0; i < length; i++)
        in[i] = i % 2 == 0 ? i : i + 1;
    clear(out);

    Pipeline p(length);

    p.stage(WINDOW_FILTER([](int64_t *x){
                              return x[0] != x[1];
                          },
                          2,
                          OUTPUT(int64_t, out),
                          INPUT(int64_t, in, std::vector<int64_t>({in[length - 1] + 1}))));

    p.fetch(out);
    p.execute();

    std::vector<int64_t> cpuResult;
    cpuResult.push_back(in[0]);
    for (uint64_t i = 1; i < in.size(); i++)
        if (in[i] != in[i - 1])
            cpuResult.push_back(in[i]);

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("window filter", p.getLength(out), cpuResult.size(), out[i], cpuResult[i]) == "none");
}

TEST_CASE("RED: Reduction", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<int64_t> in(length);
    int64_t sum = 0;

    randomize(in);

    Pipeline p(length);

    p.stage(REDUCE([](int64_t *sum, int64_t *in){
                       *sum += *in;
                   },
                   REDUCE_OUT(int64_t, &sum, ([](int64_t *sum, const int64_t *in){*sum += *in;})),
                   INPUT(int64_t, in)));

    p.fetch(&sum);
    p.execute();

    int64_t cpuSum = std::accumulate(in.begin(), in.end(), 0UL);

    OPEN_TEST_RESULT_FILE();
    REQUIRE(sum == cpuSum);
}

TEST_CASE("SCAN: Inclusive Prefix Sum", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<int64_t> in(length);
    std::vector<int64_t> scanOut(length, 0);  // seeded with identity (0 for +)

    for (uint64_t i = 0; i < length; i++)
        in[i] = static_cast<int64_t>(i);

    Pipeline p(length);

    p.stage(SCAN(([](int64_t *acc, int64_t *x){ *acc = *acc + *x; }),
                 INOUT(int64_t, scanOut),
                 INPUT(int64_t, in)));

    p.fetch(scanOut);
    p.execute();

    // Reference: serial inclusive scan
    std::vector<int64_t> cpuResult(length);
    cpuResult[0] = in[0];
    for (uint64_t i = 1; i < length; i++)
        cpuResult[i] = cpuResult[i - 1] + in[i];

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("scan", length, length, scanOut[i], cpuResult[i]) == "none");
}

TEST_CASE("HST-S: Image Histogram (Small)", "[workload]")
{
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<uint32_t> in(length);
    std::vector<uint32_t> hist(256);

    randomize(in, 4095);
    clear(hist);

    Pipeline p(length);

    p.stage(REDUCE([](uint32_t *hist, uint32_t *in){
                       hist[(*in * 256) >> 12]++;
                   },
                   REDUCE_OUT(uint32_t, hist, ([](uint32_t *out, const uint32_t *in){
                       for (uint32_t i = 0; i < 256; i++)
                           out[i] += in[i];
                   }), 256 * sizeof(uint32_t)),
                   INPUT(uint32_t, in)));

    p.fetch(hist);
    p.execute();

    std::vector<uint32_t> cpuHist(256);
    clear(cpuHist);
    for (auto pixel : in)
        cpuHist[(pixel * 256) >> 12]++;

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("reduce", 256, 256, hist[i], cpuHist[i]) == "none");
}

TEST_CASE("LINREG: Linear Regression (1 step)", "[workload]")
{
    // One linear-regression gradient step (same formulation as SimplePIM's
    // linear_reg benchmark): grad[i] = sum over points of (feat[i] * e) >> 8, with
    // e = dot(feat, weights) - (label << 5). The weights are a broadcast PARAMETER.
    const uint64_t length = 256 * 1024;
    const int dim = 10;

    std::vector<LinRegPoint> points(length);
    std::vector<int32_t> weights(dim, 0);
    LinRegGrad gradResult{};

    for (uint64_t i = 0; i < length; i++)
    {
        for (int j = 0; j < dim; j++)
            points[i].feat[j] = static_cast<int32_t>((i + j) % 10);
        points[i].label = static_cast<int32_t>(i % 10);
    }

    Pipeline p(length);
    p.addHeader(testsStructsHeaderBinData);

    p.stage(REDUCE(([](struct LinRegGrad *grad, struct LinRegPoint *in, int32_t *weights){
                       int64_t dot = 0;
                       for (int i = 0; i < 10; i++)
                           dot += in->feat[i] * weights[i];
                       int64_t e = dot - (in->label << 5);
                       for (int i = 0; i < 10; i++)
                           grad->g[i] += (in->feat[i] * e) >> 8;
                   }),
                   REDUCE_OUT(struct LinRegGrad, &gradResult,
                              ([](struct LinRegGrad *out, const struct LinRegGrad *in){
                                  for (int i = 0; i < 10; i++)
                                      out->g[i] += in->g[i];
                              })),
                   INPUT(struct LinRegPoint, points),
                   PARAMETER(int32_t, weights.data(), sizeof(int32_t) * 10)));

    p.fetch(&gradResult);
    p.execute();

    LinRegGrad cpuGrad{};
    for (uint64_t i = 0; i < length; i++)
    {
        int64_t dot = 0;
        for (int j = 0; j < dim; j++)
            dot += points[i].feat[j] * weights[j];
        int64_t e = dot - (points[i].label << 5);
        for (int j = 0; j < dim; j++)
            cpuGrad.g[j] += (points[i].feat[j] * e) >> 8;
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("reduce", 10, 10, gradResult.g[i], cpuGrad.g[i]) == "none");
}

TEST_CASE("KMEANS: K-Means assignment (1 step)", "[workload]")
{
    // One K-Means assignment step (same formulation as SimplePIM's kmeans benchmark):
    // each point is added to the bucket of its nearest centroid (squared Euclidean
    // distance). The centroids are a broadcast PARAMETER.
    const uint64_t length = 256 * 1024;
    const int dim = 10;
    const int k = 10;

    std::vector<KMeansPoint> points(length);
    std::vector<int32_t> centroids(k * dim, 0);
    KMeansResult result{};

    for (uint64_t i = 0; i < length; i++)
        for (int j = 0; j < dim; j++)
            points[i].feat[j] = static_cast<int32_t>((i + j) % 10);

    Pipeline p(length);
    p.addHeader(testsStructsHeaderBinData);

    p.stage(REDUCE(([](struct KMeansResult *acc, struct KMeansPoint *in, int32_t *centroids){
                       uint32_t best = 0;
                       uint64_t shortest = UINT64_MAX;
                       for (int c = 0; c < 10; c++)
                       {
                           uint64_t dist = 0;
                           for (int j = 0; j < 10; j++)
                           {
                               int32_t d = in->feat[j] - centroids[c * 10 + j];
                               dist += (uint64_t)(d * d);
                           }
                           if (dist < shortest) { shortest = dist; best = c; }
                       }
                       acc->b[best].count += 1;
                       for (int j = 0; j < 10; j++)
                           acc->b[best].sum[j] += in->feat[j];
                   }),
                   REDUCE_OUT(struct KMeansResult, &result,
                              ([](struct KMeansResult *out, const struct KMeansResult *in){
                                  for (int c = 0; c < 10; c++)
                                  {
                                      out->b[c].count += in->b[c].count;
                                      for (int j = 0; j < 10; j++)
                                          out->b[c].sum[j] += in->b[c].sum[j];
                                  }
                              })),
                   INPUT(struct KMeansPoint, points),
                   PARAMETER(int32_t, centroids.data(), sizeof(int32_t) * k * dim)));

    p.fetch(&result);
    p.execute();

    KMeansResult cpu{};
    for (uint64_t i = 0; i < length; i++)
    {
        uint32_t best = 0;
        uint64_t shortest = UINT64_MAX;
        for (int c = 0; c < k; c++)
        {
            uint64_t dist = 0;
            for (int j = 0; j < dim; j++)
            {
                int32_t d = points[i].feat[j] - centroids[c * dim + j];
                dist += (uint64_t)(d * d);
            }
            if (dist < shortest) { shortest = dist; best = c; }
        }
        cpu.b[best].count += 1;
        for (int j = 0; j < dim; j++)
            cpu.b[best].sum[j] += points[i].feat[j];
    }

    OPEN_TEST_RESULT_FILE();
    for (int c = 0; c < k; c++)
    {
        REQUIRE(result.b[c].count == cpu.b[c].count);
        REQUIRE(CHECK_RESULT("reduce", 10, 10, result.b[c].sum[i], cpu.b[c].sum[i]) == "none");
    }
}

TEST_CASE("LOGREG: Logistic Regression (1 step)", "[workload]")
{
    // One logistic-regression gradient step (same formulation as SimplePIM's
    // log_reg benchmark): grad[i] = sum over points of e * feat[i], with
    // e = sigmoid(dot(feat, weights)) - label. The weights are a broadcast PARAMETER.
    const uint64_t length = 256 * 1024;
    const int dim = 10;

    std::vector<LogRegPoint> points(length);
    std::vector<float> weights(dim, 0.0f);
    LogRegGrad gradResult{};

    for (uint64_t i = 0; i < length; i++)
    {
        for (int j = 0; j < dim; j++)
            points[i].feat[j] = static_cast<float>((i + j) % 10);
        points[i].label = static_cast<float>(i % 10);
    }

    Pipeline p(length);
    p.addHeader(testsStructsHeaderBinData);

    p.stage(REDUCE(([](struct LogRegGrad *grad, struct LogRegPoint *in, float *weights){
                       float dot = 0;
                       for (int i = 0; i < 10; i++)
                           dot += in->feat[i] * weights[i];
                       float s;
                       if (dot >= 15) s = 1.0f;
                       else if (dot <= -15) s = 0.0f;
                       else if (dot == 0.0f) s = 0.5f;
                       else {
                           float sum = 1.0f, temp = 1.0f;
                           for (uint32_t i = 1; i < 101; i++) { temp = temp * (-dot) / i; sum = sum + temp; }
                           s = 1.0f / (1.0f + sum);
                       }
                       float e = s - in->label;
                       for (int i = 0; i < 10; i++)
                           grad->g[i] += e * in->feat[i];
                   }),
                   REDUCE_OUT(struct LogRegGrad, &gradResult,
                              ([](struct LogRegGrad *out, const struct LogRegGrad *in){
                                  for (int i = 0; i < 10; i++)
                                      out->g[i] += in->g[i];
                              })),
                   INPUT(struct LogRegPoint, points),
                   PARAMETER(float, weights.data(), sizeof(float) * 10)));

    p.fetch(&gradResult);
    p.execute();

    LogRegGrad cpuGrad{};
    for (uint64_t i = 0; i < length; i++)
    {
        float dot = 0;
        for (int j = 0; j < dim; j++)
            dot += points[i].feat[j] * weights[j];
        float s;
        if (dot >= 15) s = 1.0f;
        else if (dot <= -15) s = 0.0f;
        else if (dot == 0.0f) s = 0.5f;
        else {
            float sum = 1.0f, temp = 1.0f;
            for (uint32_t t = 1; t < 101; t++) { temp = temp * (-dot) / t; sum = sum + temp; }
            s = 1.0f / (1.0f + sum);
        }
        float e = s - points[i].label;
        for (int j = 0; j < dim; j++)
            cpuGrad.g[j] += e * points[i].feat[j];
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("reduce", 10, 10, gradResult.g[i], cpuGrad.g[i]) == "none");
}

TEST_CASE("BS: Binary Search", "[workload]")
{
    // Binary search of many queries in a sorted array. The array is a LOOKUP
    // input, stored once in every DPU's MRAM and read on demand; the queries are
    // sharded across DPUs and tasklets by a MAP stage.
    const uint64_t arrayLen   = 1024 * 1024;
    const uint64_t numQueries = arrayLen / 8;    // Same ratio as PrIM BS
    std::vector<int64_t> sortedArray(arrayLen);
    std::vector<int64_t> queries(numQueries);
    std::vector<int64_t> results(numQueries);

    // Same inputs as PrIM BS: a sorted array and consecutive queries.
    for (uint64_t i = 0; i < arrayLen; i++)   sortedArray[i] = static_cast<int64_t>(i) + 1;
    for (uint64_t i = 0; i < numQueries; i++) queries[i]     = static_cast<int64_t>(i);

    uint64_t len = arrayLen;
    Pipeline p(numQueries);
    p.stage(MAP([](int64_t *query, LOOKUP_PTR(int64_t) array, int64_t *result, uint64_t *len){
                    int64_t lo = 0;
                    int64_t hi = (int64_t)(*len) - 1;
                    int64_t found = -1;
                    while (lo <= hi) {
                        int64_t mid = lo + ((hi - lo) >> 1);
                        int64_t v = LOOKUP(array, mid);
                        if (v == *query) { found = mid; break; }
                        else if (v < *query) lo = mid + 1;
                        else hi = mid - 1;
                    }
                    *result = found;
                },
                INPUT(int64_t, queries),
                LOOKUP_INPUT(int64_t, sortedArray, sizeof(int64_t) * arrayLen),
                OUTPUT(int64_t, results),
                PARAMETER(uint64_t, &len)));

    p.fetch(results);
    p.execute();

    // Reference: serial binary search
    std::vector<int64_t> refResults(numQueries);
    for (uint64_t i = 0; i < numQueries; i++)
    {
        int64_t lo = 0, hi = (int64_t)arrayLen - 1, res = -1;
        while (lo <= hi)
        {
            int64_t mid = lo + ((hi - lo) >> 1);
            if (sortedArray[mid] == queries[i]) { res = mid; break; }
            else if (sortedArray[mid] < queries[i]) lo = mid + 1;
            else hi = mid - 1;
        }
        refResults[i] = res;
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("map", numQueries, numQueries, results[i], refResults[i]) == "none");
}

TEST_CASE("TS: Matrix Profile (sliding dot product)", "[workload]")
{
    // Sliding-window dot product of a time series T with a query Q of length m,
    // as a single WINDOW stage: score[i] = <T[i..i+m-1], Q>. Q and m are broadcast
    // PARAMETERs.
    const uint64_t tsLength = 65536 * lengthScale;
    const uint32_t queryLength = 64;
    std::vector<int32_t> T(tsLength);
    std::vector<int32_t> Q(queryLength);
    std::vector<int32_t> scores(tsLength, 0);
    std::vector<int32_t> padding(queryLength - 1, 0);

    randomize(T);
    randomize(Q);

    uint32_t m = queryLength;
    Pipeline p(tsLength);
    p.stage(WINDOW([](int32_t *window, int32_t *Q, int32_t *score, uint32_t *m){
                       int32_t s = 0;
                       for (uint32_t i = 0; i < *m; i++) s += window[i] * Q[i];
                       *score = s;
                   },
                   queryLength,
                   INPUT(int32_t, T, padding),
                   PARAMETER(int32_t, Q, sizeof(int32_t) * queryLength),
                   OUTPUT(int32_t, scores),
                   PARAMETER(uint32_t, &m)));

    p.fetch(scores);
    p.execute();

    // Reference: serial sliding-window dot product
    const uint64_t numWindows = tsLength - queryLength + 1;
    std::vector<int32_t> refScores(numWindows);
    for (uint64_t i = 0; i < numWindows; i++)
    {
        int32_t s = 0;
        for (uint32_t j = 0; j < queryLength; j++) s += T[i + j] * Q[j];
        refScores[i] = s;
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("window", numWindows, numWindows, scores[i], refScores[i]) == "none");
}

TEST_CASE("MLP: 3-Layer Multi-Layer Perceptron", "[workload]")
{
    // Three chained GEMV layers with square matrices. Each layer is its own
    // Pipeline; the output of one layer is the PARAMETER vector of the next.
    const uint32_t n = 512;                 // square matrix dimension
    constexpr uint32_t NUM_LAYERS = 3;
    std::vector<std::vector<int32_t>> M(NUM_LAYERS, std::vector<int32_t>(n * n));
    std::vector<int32_t> B(n);
    std::vector<int32_t> C(n);

    for (auto &m : M) randomize(m);
    randomize(B);

    // Reference computation (serial CPU, before DaPPA mutates B)
    std::vector<int32_t> refB = B;
    for (uint32_t l = 0; l < NUM_LAYERS; l++)
    {
        std::vector<int32_t> nextB(n);
        for (uint32_t i = 0; i < n; i++)
        {
            int32_t s = 0;
            for (uint32_t j = 0; j < n; j++) s += M[l][i * n + j] * refB[j];
            nextB[i] = s;
        }
        refB = nextB;
    }

    uint32_t cols = n;
    for (uint32_t l = 0; l < NUM_LAYERS; l++)
    {
        clear(C);
        Pipeline p(n * n);
        p.stage(GROUP([](int32_t *M, int32_t *a, int32_t *b, uint32_t *cols){
                          int32_t res = 0;
                          for (uint32_t i = 0; i < *cols; i++) res += M[i] * a[i];
                          *b = res;
                      },
                      cols,
                      INPUT(int32_t, M[l]),
                      PARAMETER(int32_t, B, sizeof(int32_t) * cols),
                      OUTPUT(int32_t, C),
                      PARAMETER(uint32_t, &cols)));
        p.fetch(C);
        p.execute();
        std::swap(B, C);     // This layer's output is the next layer's input
    }

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("group", n, n, B[i], refB[i]) == "none");
}

TEST_CASE("HST-L: Image Histogram (Large)", "[workload]")
{
    // Same configuration as PrIM HST-L: 256 bins over 12-bit pixels. The
    // histogram is small enough for one private copy per tasklet.
    const uint64_t length = 1024 * 1024 * lengthScale;
    std::vector<uint32_t> in(length);
    std::vector<uint32_t> hist(256);

    randomize(in, 4095);     // 12-bit pixels
    clear(hist);

    Pipeline p(length);

    p.stage(REDUCE([](uint32_t *hist, uint32_t *in){
                       uint32_t bin = (*in * 256) >> 12;
                       hist[bin]++;
                   },
                   REDUCE_OUT(uint32_t, hist, ([](uint32_t *out, const uint32_t *in){
                       for (uint32_t i = 0; i < 256; i++)
                           out[i] += in[i];
                   }), 256 * sizeof(uint32_t)),
                   INPUT(uint32_t, in)));

    p.fetch(hist);
    p.execute();

    std::vector<uint32_t> cpuHist(256);
    clear(cpuHist);
    for (auto pixel : in)
        cpuHist[(pixel * 256) >> 12]++;

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("reduce", 256, 256, hist[i], cpuHist[i]) == "none");
}

TEST_CASE("HST-L-R2: Large-bin Histogram (shared banks)", "[workload]")
{
    // A 4096-bin (16 KB) histogram does not fit one private copy per tasklet in
    // WRAM, so the tasklets share a few copies, each guarded by a mutex. This
    // exercises the shared-bank reduction path.
    const uint32_t bins   = 4096;
    const uint64_t length = 512 * 1024;
    std::vector<uint32_t> in(length);
    std::vector<uint32_t> hist(bins);

    randomize(in, 4095);     // 12-bit pixels, one bin per value
    clear(hist);

    Pipeline p(length);

    p.stage(REDUCE([](uint32_t *hist, uint32_t *in){
                       uint32_t bin = (*in * 4096) >> 12;
                       hist[bin]++;
                   },
                   REDUCE_OUT(uint32_t, hist, ([](uint32_t *out, const uint32_t *in){
                       for (uint32_t i = 0; i < 4096; i++)
                           out[i] += in[i];
                   }), 4096 * sizeof(uint32_t)),
                   INPUT(uint32_t, in)));

    p.fetch(hist);
    p.execute();

    std::vector<uint32_t> cpuHist(bins);
    clear(cpuHist);
    for (auto pixel : in)
        cpuHist[(pixel * 4096) >> 12]++;

    OPEN_TEST_RESULT_FILE();
    REQUIRE(CHECK_RESULT("reduce", bins, bins, hist[i], cpuHist[i]) == "none");
}
