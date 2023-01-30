#include "cuda_utils.cuh"

// See https://stackoverflow.com/a/37569519
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

#define LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVES) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
    const size_t *rhs_strides, \
    TYPENAME *out, \
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
    TYPENAME x = lhs[lhs_i]; \
    TYPENAME y = rhs[rhs_i]; \
    TYPENAME fx; \
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
    const TYPENAME *lhs, \
    TYPENAME *grad_lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
    TYPENAME *grad_rhs, \
    const size_t *rhs_strides, \
    const TYPENAME *grad_out, \
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
    TYPENAME dfdx, dfdy; \
    DERIVATIVES \
\
    atomicAdd(grad_lhs + lhs_i, dfdx * go); \
    atomicAdd(grad_rhs + rhs_i, dfdy * go); \
}

#define BINARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DFDX, DFDY) \
    LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, fx = (FUNC);, dfdx = (DFDX); dfdy = (DFDY);)
