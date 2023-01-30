#include "cuda_utils.cuh"

extern "C" __global__ void reshape_forward_f32(
    const size_t numel,
    const float *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    float *out,
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

extern "C" __global__ void reshape_backward_f32(
    const size_t numel,
    float *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const float *grad_out,
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


extern "C" __global__ void reshape_forward_f64(
    const size_t numel,
    const double *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    double *out,
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

extern "C" __global__ void reshape_backward_f64(
    const size_t numel,
    double *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const double *grad_out,
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
