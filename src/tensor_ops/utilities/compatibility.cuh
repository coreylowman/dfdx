#include "cuda_fp16.h"

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// FIXME: the minimum compute capabilities are just guesses since the table is not specific enough

#if __CUDA_ARCH__ <= 800
__device__ __forceinline__ __half __hmax(__half a, __half b) {
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
}
__device__ __forceinline__ __half __hmin(__half a, __half b) {
    return __float2half(fminf(__half2float(a), __half2float(b)));
}
__device__ __forceinline__ __half __hmax_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}
__device__ __forceinline__ __half __hmin_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
#endif

#if __CUDA_ARCH__ < 600
// Copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// > The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
// We assume that atomicCAS for shorts has the same compute capability restrictions.
#if __CUDA_ARCH__ < 700

// Inspired by https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
// Lots of bit and pointer magic, very unsafe!
__device__ unsigned short int atomicCAS(
    unsigned short int *address,
    unsigned short int compare,
    unsigned short int val
) {
    // Read the two shorts that make up the 32 bit int at the aligned memory location.
    unsigned int* aligned_address = (unsigned int*) ((size_t) address & ~2);
    // unsigned int my_short = *address;
    unsigned int other_short = *(unsigned short int*) ((size_t) address ^ 2);
    // Replace my_short with value in the integer.
    const unsigned int value = val;
    const bool aligned = ((size_t) address & 2) == 0;
    unsigned int new_whole = aligned ? (other_short << 16) | value : (value << 16) | other_short;
    
    unsigned int old = atomicCAS(aligned_address, (unsigned int) compare, new_whole);
    return aligned ? old & (0xffff) : old >> 16;
}

__device__ __half atomicAdd(__half* address, __half val) {
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(val + __ushort_as_half(assumed)));
    } while (assumed != old);
    return __ushort_as_half(old);
}

#endif
