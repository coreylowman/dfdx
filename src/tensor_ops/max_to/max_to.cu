#include "cuda_utils.cuh"

// Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
    unsigned short int* casted_address = (short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __hmax(val, __ushort_as_half(assumed))); // __hmax_nan
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
}

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
    const size_t numel,
    const size_t chunk_len,
    const T data,
    T* out
) {
    __shared__ T buf[1024];
    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_i = threadIdx.x;
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
            buf[block_i] = maxg(buf[block_i], buf[block_i_2]);
        }

        __syncthreads();
    }

    if (block_i == chunk_start) {
        atomicMaxf(out + i / chunk_len, buf[block_i]);
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
    chunk_max(numel, chunk_len, inp[inp_i], out);
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

MAX(__half, max_to_fwd_f16, max_to_bwd_f16);
MAX(float, max_to_fwd_f32, max_to_bwd_f32);
MAX(double, max_to_fwd_f64, max_to_bwd_f64);
