#include "cuda_utils.cuh"

// Efficiently computes the sum of each chunk in "data" of size chunk_len, and
// stores the sums in out[i / chunk_len]
template<typename T>
__device__ void chunk_sum(
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
            buf[block_i] += buf[block_i_2];
        }

        __syncthreads();
    }

    if (block_i == chunk_start) {
        atomicAdd(out + i / chunk_len, buf[block_i]);
    }
}

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
extern "C" __global__ void sum_to_forward_f32(
    const size_t numel,
    const size_t num_dims,
    const float elems_per_thread,
    const size_t chunk_len,
    const float *inp,
    const size_t *dims,
    const size_t *strides,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_sum(numel, chunk_len, inp[inp_i] * elems_per_thread, out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_backward_f32(
    const size_t numel,
    const size_t num_dims,
    const float elems_per_thread,
    const size_t *dims,
    float *grad_inp,
    const size_t *inp_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);
    auto tmp = grad_out[out_i];

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with forward.
    grad_inp[inp_i] += tmp * elems_per_thread;
}

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
extern "C" __global__ void sum_to_forward_f64(
    const size_t numel,
    const size_t num_dims,
    const double elems_per_thread,
    const size_t chunk_len,
    const double *inp,
    const size_t *dims,
    const size_t *strides,
    double *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_sum(numel, chunk_len, inp[inp_i] * elems_per_thread, out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_backward_f64(
    const size_t numel,
    const size_t num_dims,
    const double elems_per_thread,
    const size_t *dims,
    double *grad_inp,
    const size_t *inp_strides,
    const double *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);
    auto tmp = grad_out[out_i];

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with forward.
    grad_inp[inp_i] += tmp * elems_per_thread;
}
