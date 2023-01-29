#include "cuda_utils.cuh"

extern "C" __global__ void choose_forward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const bool *cond,
    const size_t *cond_strides,
    const float *lhs,
    const size_t *lhs_strides,
    const float *rhs,
    const size_t *rhs_strides,
    float *out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(out_i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(out_i, num_dims, dims, rhs_strides);
    unsigned int cond_i = get_strided_index(out_i, num_dims, dims, cond_strides);

    out[out_i] = cond[cond_i] ? lhs[lhs_i] : rhs[rhs_i];
}

extern "C" __global__ void choose_backward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const bool *cond,
    const size_t *cond_strides,
    float *grad_lhs,
    const size_t *lhs_strides,
    float *grad_rhs,
    const size_t *rhs_strides,
    const float *grad_out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(out_i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(out_i, num_dims, dims, rhs_strides);
    unsigned int cond_i = get_strided_index(out_i, num_dims, dims, cond_strides);

    auto go = grad_out[out_i];
    float* out_loc = cond[cond_i] ? grad_lhs + lhs_i : grad_rhs + rhs_i;

    atomicAdd(out_loc, go);
}
