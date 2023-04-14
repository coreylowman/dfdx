#include "cuda_fp16.h"

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

__device__ unsigned int restrided(
    const unsigned int strided_i,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t *new_strides
) {
    unsigned int idx = 0;
    for (int d = 0; d < num_dims; d++) {
        idx += (strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d]) * new_strides[d];
    }
    return idx;
}

// Efficiently computes the sum of each chunk in "data" of size chunk_len, and
// stores the sums in out[i / chunk_len]
template<typename T>
__device__ void chunk_sum(
    const size_t chunk_len,
    T data,
    T* out
) {
    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_i = i % warpSize;

    // Fall back to atomicAdd if chunk_len is small to reduce overhead
    if (chunk_len <= 4) {
        atomicAdd(out + i / chunk_len, data);
        return;
    }

    unsigned int chunk_i = i % chunk_len;
    unsigned int chunk_start = max((int)(warp_i - chunk_i), 0);
    unsigned int chunk_end = min((unsigned int)(warp_i + chunk_len - chunk_i), warpSize);

    unsigned int mask = (1 << chunk_end) - (1 << chunk_start);
    unsigned int tail = chunk_end - warp_i;
    T tmp;

    for (unsigned int j = 16; j > 0; j /= 2) {
        // get data from thread (warp_i + j)
        tmp = __shfl_down_sync(mask, data, j);
        // optimized version of (warp_i + j < chunk_end) 
        if (j < tail) {
            data += tmp;
        }
    }

    if (warp_i == chunk_start) {
        atomicAdd(out + i / chunk_len, data);
    }
}

extern "C" __global__ void fill_with_f32(float *buf, float value, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    buf[i] = value;
}

extern "C" __global__ void fill_with_f64(double *buf, double value, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    buf[i] = value;
}

__device__ __forceinline__ float sqrtg(float a) { return sqrtf(a); }
__device__ __forceinline__ double sqrtg(double a) { return sqrt(a); }
__device__ __forceinline__ float powg(float a, float b) { return powf(a, b); }
__device__ __forceinline__ double powg(double a, double b) { return pow(a, b); }
__device__ __forceinline__ float tanhg(float a) { return tanhf(a); }
__device__ __forceinline__ double tanhg(double a) { return tanh(a); }
__device__ __forceinline__ float maxg(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double maxg(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ float ming(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double ming(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ float logg(float a) { return logf(a); }
__device__ __forceinline__ double logg(double a) { return log(a); }
__device__ __forceinline__ float expg(float a) { return expf(a); }
__device__ __forceinline__ double expg(double a) { return exp(a); }
__device__ __forceinline__ float absg(float a) { return fabsf(a); }
__device__ __forceinline__ double absg(double a) { return fabs(a); }
__device__ __forceinline__ float copysigng(float a, float b) { return copysignf(a, b); }
__device__ __forceinline__ double copysigng(double a, double b) { return copysign(a, b); }
