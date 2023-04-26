#include "cuda_fp16.h"

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// FIXME: the minimum compute capabilities are just guesses since the table is not specific enough

#if __CUDA_ARCH__ < 600
__device__ __forceinline__ __half __hmax(__half a, __half b) {
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
}
__device__ __forceinline__ __half __hmin(__half a, __half b) {
    return __float2half(fminf(__half2float(a), __half2float(b)));
}
#endif

#if __CUDA_ARCH__ < 700
__device__ __forceinline__ __half __hmax_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}
__device__ __forceinline__ __half __hmin_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
#endif

#if __CUDA_ARCH__ < 600
// Copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
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


#if __CUDA_ARCH__ < 700
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
// Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
__device__ __half atomicAdd(__half *address, __half val) {
    unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t) address & 2;
    do {
        assumed = old;
        unsigned int hsum;
        hsum = unaligned ? (old >> 16) : (old & 0xffff);
        hsum = __half_as_ushort(__ushort_as_half(hsum) + val); 
        old = atomicCAS(address_as_ui, assumed,
            unaligned ? (old & 0xffff) | (hsum << 16) : (old & 0xffff0000) | hsum
        );

   } while (assumed != old);
   return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
}
#endif
