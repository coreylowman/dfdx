#include "cuda_utils.cuh"

template<typename T>
__device__ void slice_fwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t offset,
    const T *inp,
    T *out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int inp_i = offset + get_strided_index(out_i, num_dims, dims, strides);
    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void slice_bwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t offset,
    T *grad_inp,
    const T *out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int inp_i = offset + get_strided_index(out_i, num_dims, dims, strides);
    // TODO (maybe): use chunk_sum to speed this up 
    atomicAdd(grad_inp + inp_i, out[out_i]);
}

#define SLICE(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t offset, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    slice_fwd(numel, num_dims, dims, strides, offset, inp, out); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t offset, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    slice_bwd(numel, num_dims, dims, strides, offset, grad_inp, grad_out); \
}

SLICE(float, slice_fwd_f32, slice_bwd_f32);
SLICE(double, slice_fwd_f64, slice_bwd_f64);
