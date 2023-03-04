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

__device__ unsigned int get_unstrided_index(
    const unsigned int strided_i,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int idx = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        idx *= dims[d];
        idx += strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d];
    }
    return idx;
}

// Sourced from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
// Input must be less than or equal to 2 ^ 16
// used in reductions
__device__ __forceinline__ unsigned int next_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v++;
    return v;
}

// Efficiently computes the sum of each chunk in "data" of size chunk_len, and
// stores the sums in out[i / chunk_len]
template<typename T>
__device__ void chunk_sum(
    const size_t chunk_len,
    const T data,
    T* out
) {
    __shared__ T buf[1024];

    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_i = threadIdx.x;

    // Fall back to atomicAdd if chunk_len is small to reduce overhead
    if (chunk_len <= 2) {
        atomicAdd(out + i / chunk_len, data);
        return;
    }
    buf[block_i] = data;

    unsigned int chunk_i = i % chunk_len;
    unsigned int chunk_start = max((int)(block_i - chunk_i), 0);
    unsigned int chunk_end = min((unsigned int)(block_i + chunk_len - chunk_i), blockDim.x);

    chunk_i = block_i - chunk_start;

    size_t max_chunk_len = min(chunk_end - chunk_start, blockDim.x);
    size_t incr = next_power_of_two(max_chunk_len) >> 1;

    __syncthreads();

    // Uses sequential addressing as discussed in
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    for (; incr > 0; incr >>= 1) {
        unsigned int block_i_2 = block_i + incr;

        if (block_i_2 < chunk_end && chunk_i < incr) {
            // This is sound because __syncthreads and the conditions above
            // ensure that no data races occur
            buf[block_i] += buf[block_i_2];
        }

        __syncthreads();
    }

    if (block_i == chunk_start) {
        atomicAdd(out + i / chunk_len, buf[block_i]);
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
