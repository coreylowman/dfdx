#include "cuda_utils.cuh"

template<typename T>
__device__ void reshape_fwd(
    const size_t numel,
    const T *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    T *out,
    const size_t out_num_dims,
    const size_t *out_dims,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void reshape_bwd(
    const size_t numel,
    T *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const T *grad_out,
    const size_t out_num_dims,
    const size_t *out_dims,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}

#define RESHAPE(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const TYPENAME *inp, \
    const size_t inp_num_dims, \
    const size_t *inp_dims, \
    const size_t *inp_strides, \
    TYPENAME *out, \
    const size_t out_num_dims, \
    const size_t *out_dims, \
    const size_t *out_strides \
) { \
    reshape_fwd(numel, inp, inp_num_dims, inp_dims, inp_strides, out, out_num_dims, out_dims, out_strides); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    TYPENAME *grad_inp, \
    const size_t inp_num_dims, \
    const size_t *inp_dims, \
    const size_t *inp_strides, \
    const TYPENAME *grad_out, \
    const size_t out_num_dims, \
    const size_t *out_dims, \
    const size_t *out_strides \
) { \
    reshape_bwd(numel, grad_inp, inp_num_dims, inp_dims, inp_strides, grad_out, out_num_dims, out_dims, out_strides); \
}

RESHAPE(float, reshape_fwd_f32, reshape_bwd_f32);
RESHAPE(double, reshape_fwd_f64, reshape_bwd_f64);
