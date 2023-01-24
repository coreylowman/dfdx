#include "cuda_utils.cuh"

struct HuberErrorOp {
    float delta;
};

extern "C" __global__ void huber_error_forward(
    const HuberErrorOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    const size_t *lhs_strides,
    const float *rhs,
    const size_t *rhs_strides,
    float *out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    float a = lhs[lhs_i] - rhs[rhs_i];

    if (fabsf(a) < op.delta) {
        out[out_i] = a * a * 0.5;
    } else {
        out[out_i] = op.delta * (fabsf(a) - 0.5 * op.delta);
    }
}

extern "C" __global__ void huber_error_backward(
    const HuberErrorOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    float *grad_lhs,
    const size_t *lhs_strides,
    const float *rhs,
    float *grad_rhs,
    const size_t *rhs_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    auto a = lhs[lhs_i] - rhs[rhs_i];
    auto go = grad_out[out_i];

    float dfdx, dfdy;

    if (a == 0.0) {
        dfdx = 0.0;
    } else if (fabsf(a) < op.delta) {
        dfdx = a;
    } else {
        dfdx = copysignf(op.delta, a);
    }

    dfdy = -dfdx;

    atomicAdd(grad_lhs + lhs_i, dfdx * go);
    atomicAdd(grad_rhs + rhs_i, dfdy * go);
}
