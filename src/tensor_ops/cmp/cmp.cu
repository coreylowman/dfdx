#include "cuda_utils.cuh"

extern "C" __global__ void eq_forward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    const size_t *lhs_strides,
    const float *rhs,
    const size_t *rhs_strides,
    bool *out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    out[out_i] = lhs[lhs_i] == rhs[rhs_i];
}
