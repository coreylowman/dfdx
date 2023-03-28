#include "cuda_utils.cuh"

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
template<typename T>
__device__ void sum_to_fwd(
    const size_t numel,
    const size_t num_dims,
    const T elems_per_thread,
    const size_t chunk_len,
    const T *inp,
    const size_t *dims,
    const size_t *strides,
    T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

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
    const size_t *dims,
    T *grad_inp,
    const size_t *inp_strides,
    const T *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int out_i = restrided(inp_i, num_dims, dims, inp_strides, out_strides);
    T tmp = grad_out[out_i];

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with fwd.
    grad_inp[inp_i] += tmp * elems_per_thread;
}

#define SUM(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const TYPENAME elems_per_thread, \
    const size_t chunk_len, \
    const TYPENAME *inp, \
    const size_t *dims, \
    const size_t *strides, \
    TYPENAME *out \
) { \
    sum_to_fwd(numel, num_dims, elems_per_thread, chunk_len, inp, dims, strides, out); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const TYPENAME elems_per_thread, \
    const size_t *dims, \
    TYPENAME *grad_inp, \
    const size_t *inp_strides, \
    const TYPENAME *grad_out, \
    const size_t *out_strides \
) { \
    sum_to_bwd(numel, num_dims, elems_per_thread, dims, grad_inp, inp_strides, grad_out, out_strides); \
}

SUM(float, sum_to_fwd_f32, sum_to_bwd_f32);
SUM(double, sum_to_fwd_f64, sum_to_bwd_f64);
