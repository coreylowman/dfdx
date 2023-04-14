#include "cuda_utils.cuh"

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));        
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMaxf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int *)addr, __double_as_longlong(value)));
    }
}

// Efficiently computes the max of each chunk in "data" of size chunk_len, and
// stores the maximums in out[i / chunk_len]
template<typename T>
__device__ void chunk_max(
    const size_t chunk_len,
    T data,
    T* out
) {
    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_i = i % warpSize;

    // Fall back to atomicMaxf if chunk_len is small to reduce overhead
    if (chunk_len <= 4) {
        atomicMaxf(out + i / chunk_len, data);
        return;
    }

    unsigned int chunk_i = i % chunk_len;
    unsigned int chunk_start = max((int)(warp_i - chunk_i), 0);
    unsigned int chunk_end = min((unsigned int)(warp_i + chunk_len - chunk_i), warpSize);

    unsigned int tail = chunk_end - warp_i;
    T tmp;

    for (unsigned int j = 16; j > 0; j /= 2) {
        // get data from thread (warp_i + j)
        tmp = __shfl_down_sync(SHFL_MASK, data, j);
        // optimized version of (warp_i + j < chunk_end) 
        if (j < tail) {
            data = max(data, tmp);
        }
    }

    if (warp_i == chunk_start) {
        atomicMaxf(out + i / chunk_len, data);
    }
}

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
template<typename T>
__device__ void max_to_fwd(
    const size_t numel,
    const size_t num_dims,
    const size_t chunk_len,
    const size_t *info,
    const T *inp,
    T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_max(chunk_len, inp[inp_i], out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
template<typename T>
__device__ void max_to_bwd(
    const size_t numel,
    const size_t num_dims,
    const T elems_per_thread,
    const size_t *info,
    const T *inp,
    T *grad_inp,
    const T *out,
    const T *grad_out
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    const size_t *dims = info;
    const size_t *inp_strides = info + num_dims;
    const size_t *out_strides = info + 2 * num_dims;

    unsigned int out_i = restrided(inp_i, num_dims, dims, inp_strides, out_strides);

    const T mask = static_cast<T>(inp[inp_i] == out[out_i]);
    grad_inp[inp_i] += mask * grad_out[out_i] * elems_per_thread;
}

#define MAX(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t chunk_len, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    max_to_fwd(numel, num_dims, chunk_len, info, inp, out); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const TYPENAME elems_per_thread, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *out, \
    const TYPENAME *grad_out \
) { \
    max_to_bwd(numel, num_dims, elems_per_thread, info, inp, grad_inp, out, grad_out); \
}

MAX(float, max_to_fwd_f32, max_to_bwd_f32);
MAX(double, max_to_fwd_f64, max_to_bwd_f64);
