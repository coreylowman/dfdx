#include "cuda_utils.cuh"

#define CMP_OP(FORWARD, SCALAR_FORWARD, SYMBOL) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const float *lhs, \
    const size_t *lhs_strides, \
    const float *rhs, \
    const size_t *rhs_strides, \
    bool *out, \
    const size_t *out_strides \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
    out[out_i] = lhs[lhs_i] SYMBOL rhs[rhs_i]; \
} \
\
extern "C" __global__ void SCALAR_FORWARD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const float *lhs, \
    const size_t *lhs_strides, \
    float scalar, \
    bool *out, \
    const size_t *out_strides \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
    out[out_i] = lhs[lhs_i] SYMBOL scalar; \
}

CMP_OP(eq_forward, scalar_eq_forward, ==)
CMP_OP(ne_forward, scalar_ne_forward, !=)
CMP_OP(gt_forward, scalar_gt_forward, >)
CMP_OP(ge_forward, scalar_ge_forward, >=)
CMP_OP(lt_forward, scalar_lt_forward, <)
CMP_OP(le_forward, scalar_le_forward, <=)
