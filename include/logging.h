#ifndef DAPPA_LOGGING_H
#define DAPPA_LOGGING_H

// Not <cstdio> as it has to compile as C code as well for the DPUs
#include <stdio.h>

// List of possible verbosity values
#define VERBOSITY_QUIET 0
#define VERBOSITY_ERROR 1
#define VERBOSITY_WARN 2
#define VERBOSITY_INFO 3
#define VERBOSITY_DEBUG 4
#define VERBOSITY_DEBUG_VERBOSE 5
#define VERBOSITY_DEBUG_ALL 6

// Adjust desired verbosity here
#define VERBOSITY VERBOSITY_QUIET

// Enable performance measurements
//#define MEASURE_PERFORMANCE_HOST
//#define MEASURE_PERFORMANCE_DPU
//#define MEASURE_PERFORMANCE_NO_HEADERS

#define ERROR_HOST(msg, ...)
#define WARN_HOST(msg, ...)
#define INFO_HOST(msg, ...)
#define DEBUG_HOST(msg, ...)
#define DEBUG_HOST_V(msg, ...)
#define DEBUG_HOST_VV(msg, ...)

#define ERROR_DPU(msg, ...)
#define WARN_DPU(msg, ...)
#define INFO_DPU(msg, ...)
#define DEBUG_DPU(msg, ...)
#define DEBUG_DPU_V(msg, ...)
#define DEBUG_DPU_VV(msg, ...)
#define TIME_DPU(name)


// --- Helpers ------------

// General output macros
#define DAPPA_PRINT(tag, msg, ...) printf(tag " " msg "\n\x1B[0m" __VA_OPT__(,) __VA_ARGS__)

#define DAPPA_HOST_PRINT(msg, ...)    do { DAPPA_PRINT("\x1B[32m[host]", msg __VA_OPT__(,) __VA_ARGS__); fflush(stdout); } while(0)
#define DAPPA_TASKLET_PRINT(msg, ...) DAPPA_PRINT("\x1B[31m[task %d]", msg __VA_OPT__(,) __VA_ARGS__)

// Tasklet specific macros with the option to filter out the output of a single tasklet
#ifdef TASKLET_LOG_FILTER
    #define DAPPA_PRINT_DPU_FILTERED(msg, ...) if (taskletIndex == TASKLET_LOG_FILTER) DAPPA_TASKLET_PRINT(msg, taskletIndex __VA_OPT__(,) __VA_ARGS__)
#else
    #define DAPPA_PRINT_DPU_FILTERED(msg, ...) DAPPA_TASKLET_PRINT(msg, taskletIndex __VA_OPT__(,) __VA_ARGS__)
#endif

// Enabling macros depending on the verbosity
#if VERBOSITY >= VERBOSITY_ERROR
    #undef ERROR_HOST
    #undef ERROR_DPU
    #define ERROR_HOST(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define ERROR_DPU(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#if VERBOSITY >= VERBOSITY_WARN
    #undef WARN_HOST
    #undef WARN_DPU
    #define WARN_HOST(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define WARN_DPU(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#if VERBOSITY >= VERBOSITY_INFO
    #undef INFO_HOST
    #undef INFO_DPU
    #define INFO_HOST(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define INFO_DPU(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#if VERBOSITY >= VERBOSITY_DEBUG
    #undef DEBUG_HOST
    #undef DEBUG_DPU
    #define DEBUG_HOST(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define DEBUG_DPU(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#if VERBOSITY >= VERBOSITY_DEBUG_VERBOSE
    #undef DEBUG_HOST_V
    #undef DEBUG_DPU_V
    #define DEBUG_HOST_V(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define DEBUG_DPU_V(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#if VERBOSITY >= VERBOSITY_DEBUG_ALL
    #undef DEBUG_HOST_VV
    #undef DEBUG_DPU_VV
    #define DEBUG_HOST_VV(msg, ...) DAPPA_HOST_PRINT(msg, __VA_ARGS__)
    #define DEBUG_DPU_VV(msg, ...) DAPPA_PRINT_DPU_FILTERED(msg, __VA_ARGS__)
#endif

#ifdef MEASURE_PERFORMANCE_DPU
    #undef TIME_DPU
    #define TIME_DPU(name) do { DAPPA_PRINT_DPU_FILTERED("%s: %lu cycles", (name), perfcounter_get()); perfcounter_config(COUNT_CYCLES, true); } while(0)
#endif

#endif //DAPPA_LOGGING_H
