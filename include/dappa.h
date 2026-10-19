#ifndef DAPPA_H
#define DAPPA_H

// Stage types. Each macro expands to the stage arguments of Pipeline::stage():
// the stringified function (compiled into the DPU program), a host-side
// std::function, the argument tuple, the window overlap and the group size.
//   MAP(func, args...)                   one output element per input element
//   FILTER(func, args...)                keep elements for which func returns true
//   REDUCE(func, args...)                fold all elements into a REDUCE_OUT accumulator
//   SCAN(func, args...)                  inclusive scan into an INOUT vector
//   WINDOW(func, windowSize, args...)    func sees windowSize consecutive input elements
//   GROUP(func, groupSize, args...)      func consumes groupSize input elements per output
// The remaining macros combine WINDOW, GROUP and FILTER.
#define MAP(func, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__), 0, 1
#define FILTER(func, ...) #func, std::function(func), DAPPA_FILTER_EXPAND(__VA_ARGS__), 0, 1
#define REDUCE(func, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__)
#define SCAN(func, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__), DaPPA::ScanTag{}
#define WINDOW(func, windowSize, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__), (windowSize) - 1, 1
#define GROUP(func, groupSize, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__), 0, (groupSize)
#define WINDOW_FILTER(func, windowSize, ...) #func, std::function(func), DAPPA_FILTER_EXPAND(__VA_ARGS__), (windowSize) - 1, 1
#define GROUP_FILTER(func, groupSize, ...) #func, std::function(func), DAPPA_FILTER_EXPAND(__VA_ARGS__), 0, (groupSize)
#define WINDOW_GROUP(func, groupSize, windowSize, ...) #func, std::function(func), std::make_tuple(__VA_ARGS__), (windowSize) - 1, (groupSize)
#define WINDOW_GROUP_FILTER(func, groupSize, windowSize, ...) #func, std::function(func), DAPPA_FILTER_EXPAND(__VA_ARGS__), (windowSize) - 1, (groupSize)

// Argument types. T is the element type and ptr the host buffer.
//   INPUT       vector read by the stage (optional padding used past the end of the data)
//   OUTPUT      vector written by the stage
//   INOUT       vector read and written by the stage (optional padding)
//   PARAMETER   scalar broadcast to all DPUs (optional size in bytes for arrays)
//   REDUCE_OUT  scalar reduction accumulator (optional host combine function and size in bytes)
#define INPUT(T, ptr, ...) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Vector | DaPPA::ArgFlags::Input __VA_OPT__(,) __VA_ARGS__)
#define OUTPUT(T, ptr) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Vector | DaPPA::ArgFlags::Output)
#define INOUT(T, ptr, ...) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Vector | DaPPA::ArgFlags::Input | DaPPA::ArgFlags::Output | DaPPA::ArgFlags::InOut __VA_OPT__(,) __VA_ARGS__)
#define PARAMETER(T, ptr, ...) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Scalar | DaPPA::ArgFlags::Input | DaPPA::ArgFlags::Parameter __VA_OPT__(,nullptr,) __VA_ARGS__)
#define REDUCE_OUT(T, ptr, ...) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Scalar | DaPPA::ArgFlags::Input | DaPPA::ArgFlags::Output | DaPPA::ArgFlags::Reduced __VA_OPT__(,) __VA_ARGS__)
// Randomly indexed table of sizeBytes bytes, broadcast to the MRAM of every DPU. Read it
// with LOOKUP(arr, idx) and declare the stage parameter as LOOKUP_PTR(T) so the same code
// compiles for the host and the DPU.
#define LOOKUP_INPUT(T, ptr, sizeBytes) DaPPA::ArgTyped<T>(#T, ptr, DaPPA::ArgFlags::Lookup | DaPPA::ArgFlags::Input, nullptr, (sizeBytes))

// Host versions of LOOKUP and LOOKUP_PTR. The DPU versions (backed by MRAM reads) are
// defined in device_template.c.inja.
#ifndef LOOKUP
#define LOOKUP(arr, idx) ((arr)[(idx)])
#endif
#ifndef LOOKUP_PTR
#define LOOKUP_PTR(T) T*
#endif

// Optimization switches
//#define DAPPA_SAFE_CACHE_ALLOCATION  // Reduces performance, but enables access to inputs after writing an output within a stage
#define DAPPA_MAX_DPU_COUNT 16  // Maximum number of DPUs that should be used (0 = maximum available)
#define DAPPA_TARGET_CPU_LOAD 0  // Fraction of the load that should be processed by the CPU (approximate, between 0 and 1)


// Headers
#include <tuple>
#include <functional>
#include "pipeline.h"
#include "pipeline_full.h"
#include "arg_typed.h"
#include "arg_flags.h"

#endif //DAPPA_H
