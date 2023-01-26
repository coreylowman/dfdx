#include "cuda_utils.cuh"

#define LONG_BINARY_OP(FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVES) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const float *lhs, \
    const size_t *lhs_strides, \
    const float *rhs, \
    const size_t *rhs_strides, \
    float *out, \
    const size_t *out_strides \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
\
    float x = lhs[lhs_i]; \
    float y = rhs[rhs_i]; \
    float fx; \
\
    FUNC\
\
    out[out_i] = fx; \
} \
\
extern "C" __global__ void BACKWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const float *lhs, \
    float *grad_lhs, \
    const size_t *lhs_strides, \
    const float *rhs, \
    float *grad_rhs, \
    const size_t *rhs_strides, \
    const float *grad_out, \
    const size_t *out_strides \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
\
    auto x = lhs[lhs_i]; \
    auto y = rhs[rhs_i]; \
    auto go = grad_out[out_i]; \
\
    float dfdx, dfdy; \
    DERIVATIVES \
\
    atomicAdd(grad_lhs + lhs_i, dfdx * go); \
    atomicAdd(grad_rhs + rhs_i, dfdy * go); \
}

#define BINARY_OP(FORWARD, BACKWARD, OP_STRUCT, FUNC, DFDX, DFDY) \
    LONG_BINARY_OP(FORWARD, BACKWARD, OP_STRUCT, fx = (FUNC);, dfdx = (DFDX); dfdy = (DFDY);)
