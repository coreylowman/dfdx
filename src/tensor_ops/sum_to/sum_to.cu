#include "cuda_utils.cuh"

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
template<typename T>
__device__ void sum_to_fwd(
    const size_t numel,
    const size_t num_dims,
    const T elems_per_thread,
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
    chunk_sum(chunk_len, inp[inp_i] * elems_per_thread, out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
template<typename T>
__device__ void sum_to_bwd(
    const size_t numel,
    const size_t num_dims,
    const T elems_per_thread,
    const size_t *info,
    T *grad_inp,
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

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with fwd.
    grad_inp[inp_i] += grad_out[out_i] * elems_per_thread;
}

#define SUM(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const TYPENAME elems_per_thread, \
    const size_t chunk_len, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    sum_to_fwd(numel, num_dims, elems_per_thread, chunk_len, info, inp, out); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const TYPENAME elems_per_thread, \
    const size_t *info, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    sum_to_bwd(numel, num_dims, elems_per_thread, info, grad_inp, grad_out); \
}

SUM(__half, sum_to_fwd_f16, sum_to_bwd_f16);
SUM(float, sum_to_fwd_f32, sum_to_bwd_f32);
SUM(double, sum_to_fwd_f64, sum_to_bwd_f64);

__device__ void chunk_sum_amp_f16(
    const size_t chunk_len,
    const __half data,
    __half* out
) {
    __shared__ float buf[1024];

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
        __half y = buf[block_i];
        atomicAdd(out + i / chunk_len, y);
    }
}

extern "C" __global__ void sum_to_fwd_amp_f16(
    const size_t numel,
    const size_t num_dims,
    const __half elems_per_thread,
    const size_t chunk_len,
    const size_t *info,
    const __half *inp,
    __half *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_sum_amp_f16(chunk_len, inp[inp_i] * elems_per_thread, out);
}
