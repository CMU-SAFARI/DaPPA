#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <handshake.h>
#include <mutex.h>
#include <perfcounter.h>

#define TASKLET_LOG_FILTER 0
//#define TASKLET_FILTER 0

// Directly include the headers
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

    #ifndef DAPPA_SHARED_H
#define DAPPA_SHARED_H

#include <stdint.h>

struct HostData
{
    uint32_t taskletElementCount;
    uint32_t additionalElementTaskletCount;
};

#endif //DAPPA_SHARED_H


// Architectural constants
#define mramAlignment 8
#define mramMaxTransferSize 2048
typedef uint64_t MramAlignedType;

uint32_t paddedSize(uint32_t size)
{
    // Only works if mramAlignment is 8
    return (size + 7) & ~7;
}

uint32_t aligned(uint32_t size)
{
    // Only works if mramAlignment is 8
    return size & ~7;
}

uint32_t min(uint32_t one, uint32_t two)
{
    return one < two ? one : two;
}

// MRAM transfers of arbitrary size, split into chunks of at most mramMaxTransferSize bytes
void mram_write_full(void *src, __mram_ptr void *dst, uint32_t size)
{
    for (uint32_t offset = 0; size > 0; offset += mramMaxTransferSize, size -= min(size, mramMaxTransferSize))
        mram_write((uint8_t*)src + offset, (__mram_ptr uint8_t*)dst + offset, min(size, mramMaxTransferSize));
}

void mram_read_full(__mram_ptr void *src, void *dst, uint32_t size)
{
    for (uint32_t offset = 0; size > 0; offset += mramMaxTransferSize, size -= min(size, mramMaxTransferSize))
        mram_read((__mram_ptr uint8_t*)src + offset, (uint8_t*)dst + offset, min(size, mramMaxTransferSize));
}

// DPU versions of the lookup table accessors (the host versions are in dappa.h).
// LOOKUP reads the 8-byte aligned window that contains the element from MRAM and
// returns the element by value.
#define LOOKUP_PTR(T) __mram_ptr T*
#define LOOKUP(arr, idx) ({                                                       \
    __dma_aligned MramAlignedType __lk_buf[(sizeof((arr)[0]) + 2 * mramAlignment) \
                                           / sizeof(MramAlignedType)];            \
    __mram_ptr uint8_t const *__lk_p = (__mram_ptr uint8_t const *)((arr) + (idx)); \
    uint32_t __lk_off = (uint32_t)((uintptr_t)__lk_p & (mramAlignment - 1));       \
    mram_read((__mram_ptr void const *)(__lk_p - __lk_off), __lk_buf,             \
              paddedSize(__lk_off + sizeof((arr)[0])));                           \
    __typeof__((arr)[0] + (arr)[0]) __lk_v;                                       \
    __builtin_memcpy(&__lk_v, (uint8_t *)__lk_buf + __lk_off, sizeof((arr)[0]));  \
    __lk_v;                                                                       \
})

// All kernel functions
    static inline
        void
    kernelFunc0    (int32_t *a, int32_t *b, int32_t *c){ *c = *a + *b; } 


// Host inputs/outputs
__host struct HostData hostData;

    // Filter output lengths

    // Padded variables for scalars. Reduce outputs have R accumulator banks, which are
    // guarded by a mutex if R < NR_TASKLETS. Read-only scalars are stored only once to
    // save WRAM.

    // Scan state. Each launch scans in two passes:
    //  1. Every tasklet scans its contiguous region and records its total.
    //  2. Tasklet 0 computes the per-tasklet offsets and the total of the DPU
    //     (scanChunkSumArg), then every tasklet adds its offset to its region.
    // The host combines the totals of all DPUs and rounds afterwards.
    // Filter output lengths

    // Padded variables for scalars. Reduce outputs have R accumulator banks, which are
    // guarded by a mutex if R < NR_TASKLETS. Read-only scalars are stored only once to
    // save WRAM.

    // Scan state. Each launch scans in two passes:
    //  1. Every tasklet scans its contiguous region and records its total.
    //  2. Tasklet 0 computes the per-tasklet offsets and the total of the DPU
    //     (scanChunkSumArg), then every tasklet adds its offset to its region.
    // The host combines the totals of all DPUs and rounds afterwards.
    // Filter output lengths

    // Padded variables for scalars. Reduce outputs have R accumulator banks, which are
    // guarded by a mutex if R < NR_TASKLETS. Read-only scalars are stored only once to
    // save WRAM.

    // Scan state. Each launch scans in two passes:
    //  1. Every tasklet scans its contiguous region and records its total.
    //  2. Tasklet 0 computes the per-tasklet offsets and the total of the DPU
    //     (scanChunkSumArg), then every tasklet adds its offset to its region.
    // The host combines the totals of all DPUs and rounds afterwards.

// Synchronization for memory initialization
BARRIER_INIT(memInitBarrier, NR_TASKLETS);


int main()
{
    uint32_t taskletIndex = me();
    uint32_t taskletElementCount;
    uint32_t mramIndexIn;
    uint32_t mramIndexOut;
    uint32_t cacheIndex;
    uint32_t cacheIndexGroup;
    uint32_t filterCacheIndex;
    uint32_t filterMramOffset;
    uint32_t elementsLeft;
    uint32_t elementsThisBlock;
    uint32_t filterCacheOffset;
    uint32_t dpuRegionElements;
    uint32_t blockStartElems;

#ifdef MEASURE_PERFORMANCE_DPU
    perfcounter_config(COUNT_CYCLES, true);
#endif

    DEBUG_DPU("tasklet count: %d  additional: %d  size: %d", hostData.taskletElementCount, hostData.additionalElementTaskletCount, sizeof(hostData));

    // Initialize the memory if needed
    if(taskletIndex == 0)
        mem_reset();
    barrier_wait(&memInitBarrier);

    TIME_DPU("Memory init");

#ifdef TASKLET_FILTER
    if (taskletIndex != TASKLET_FILTER)
        return 0;
#endif

    taskletElementCount = hostData.taskletElementCount;
    // Special case for the very end of the data
    if (taskletIndex < hostData.additionalElementTaskletCount)
        taskletElementCount++;

        // Reset the output sizes

        // Get the correct scalar args. Reduce outputs use bank taskletIndex % R, read-only
        // scalars only have a single copy.
        // Reset the output sizes

        // Get the correct scalar args. Reduce outputs use bank taskletIndex % R, read-only
        // scalars only have a single copy.
        // Reset the output sizes

        // Get the correct scalar args. Reduce outputs use bank taskletIndex % R, read-only
        // scalars only have a single copy.

    // Allocate a WRAM cache for the args
    __dma_aligned uint8_t *cache = mem_alloc(5184);

    // Calculate the start of our MRAM block
    uint32_t mramStartIndex;
    if (taskletIndex < hostData.additionalElementTaskletCount)
        mramStartIndex = taskletIndex * taskletElementCount;
    else
        mramStartIndex = hostData.additionalElementTaskletCount * (taskletElementCount + 1) + (taskletIndex - hostData.additionalElementTaskletCount) * taskletElementCount;

    DEBUG_DPU("Tasklet start (elements: %d, stack: %d, mram: %p+%d)", taskletElementCount, check_stack(), DPU_MRAM_HEAP_POINTER, mramStartIndex);

    // Correctly type the cache MRAM pointers for every argument
        int32_t *argCache2;
        __mram_ptr int32_t *argMramPtr2 = (__mram_ptr int32_t *)
                                                                ((__mram_ptr uint8_t*)(DPU_MRAM_HEAP_POINTER) + 0);
        int32_t *argCache1;
        __mram_ptr int32_t *argMramPtr1 = (__mram_ptr int32_t *)
                                                                ((__mram_ptr uint8_t*)(DPU_MRAM_HEAP_POINTER) + 262064);
        int32_t *argCache0;
        __mram_ptr int32_t *argMramPtr0 = (__mram_ptr int32_t *)
                                                                ((__mram_ptr uint8_t*)(DPU_MRAM_HEAP_POINTER) + 524128);

    TIME_DPU("Arg init");

        // Reset memory offsets for each stage
        mramIndexIn = mramStartIndex / 1;
        mramIndexOut = mramStartIndex / 1;
        filterMramOffset = 0 + (mramStartIndex / 1) * 0;
        cacheIndex = 0;
        filterCacheIndex = 0;
        filterCacheOffset = 0;

        DEBUG_DPU("-----------------");
        DEBUG_DPU("Starting stage 0");

        // Get the correct input element count
            elementsLeft = taskletElementCount / 1 + 0 * 1 + 0;

        // Filters will fill up that value themselves

        // Initialize the base cache pointers
            argCache0 = (int32_t*)(cache + 0);
            argCache1 = (int32_t*)(cache + 1728);
            argCache2 = (int32_t*)(cache + 0);

        // Keep private reduce accumulators of up to 8 bytes in a local variable, so the
        // compiler can hold them in registers (argScalar may alias the cache). They are
        // written back to their WRAM bank after the stage.


        TIME_DPU("Stage init");

        // Interleaved indexing: tasklet t processes blocks t, t + NR_TASKLETS, ..., which keeps
        // the tasklets out of phase so their MRAM transfers overlap instead of running in lockstep
        dpuRegionElements = NR_TASKLETS * hostData.taskletElementCount + hostData.additionalElementTaskletCount;
        for (blockStartElems = taskletIndex * 432;
             blockStartElems < dpuRegionElements;
             blockStartElems += 432 * NR_TASKLETS)
        {
            elementsThisBlock = min(dpuRegionElements - blockStartElems, 432);
            mramIndexIn = blockStartElems;
            mramIndexOut = blockStartElems;

            DEBUG_DPU("Starting block (size: %d)", elementsThisBlock);

            TIME_DPU("Block init");

            // Read inputs to caches
                    DEBUG_DPU("Reading input: M+%d -> C+%d (size %d, inputOverlap %d)",
                              524128 + mramIndexIn,
                              0,
                              paddedSize(4 * (elementsThisBlock + 0)),
                              0);
                    mram_read_full(argMramPtr0 + mramIndexIn,
                                   argCache0,
                                   paddedSize(4 * (elementsThisBlock + 0)));
                    DEBUG_DPU("Reading input: M+%d -> C+%d (size %d, inputOverlap %d)",
                              262064 + mramIndexIn,
                              1728,
                              paddedSize(4 * (elementsThisBlock + 0)),
                              0);
                    mram_read_full(argMramPtr1 + mramIndexIn,
                                   argCache1,
                                   paddedSize(4 * (elementsThisBlock + 0)));

            TIME_DPU("Copy Inputs");

            // Partially unroll the per-element loop to reduce loop overhead
            #pragma unroll 8
            for (cacheIndex = 0; cacheIndex < elementsThisBlock; cacheIndex += 1)
            {
                // Capture the result of the kernel if it is a filter

                // Shared reduce accumulator banks must be locked

                // Run the kernel
                kernelFunc0(
                                argCache0 + cacheIndex
                            ,
                                argCache1 + cacheIndex
                            ,
                                argCache2 + cacheIndex
                );



                // Copy the filter output
                DEBUG_DPU_VV("--- cacheIndex = %d ---", cacheIndex);

                // Debug output
                            DEBUG_DPU_VV("Vector: %d", *(argCache0 + cacheIndex));
                            DEBUG_DPU_VV("Vector: %d", *(argCache1 + cacheIndex));
                            DEBUG_DPU_VV("Vector: %d", *(argCache2 + cacheIndex));
            }

            TIME_DPU("Process block");

            // Write outputs back to MRAM
                    DEBUG_DPU("Writing output: C+%d -> M+%d (size %d, groupSize %d)",
                              0,
                              0 + mramIndexOut,
                              aligned(4 * (elementsThisBlock / 1)),
                              1);

                    // Aligned, not padded as everything is aligned except the last tasklet, which we
                    // account for with the alignmentOverlap
                    mram_write_full(argCache2,
                                    argMramPtr2 + mramIndexOut,
                                    aligned(4 * (elementsThisBlock / 1)));

            TIME_DPU("Writing outputs");


            TIME_DPU("Writing filter output");

        }

        // Write the local reduce accumulators back to WRAM


    return 0;
}
