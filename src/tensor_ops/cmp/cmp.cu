#include "cuda_utils.cuh"

#define CMP_OP(TYPENAME, FORWARD, SCALAR_FORWARD, SYMBOL) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
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
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    TYPENAME scalar, \
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

CMP_OP(float, eq_forward_f32, scalar_eq_forward_f32, ==)
CMP_OP(float, ne_forward_f32, scalar_ne_forward_f32, !=)
CMP_OP(float, gt_forward_f32, scalar_gt_forward_f32, >)
CMP_OP(float, ge_forward_f32, scalar_ge_forward_f32, >=)
CMP_OP(float, lt_forward_f32, scalar_lt_forward_f32, <)
CMP_OP(float, le_forward_f32, scalar_le_forward_f32, <=)
CMP_OP(double, eq_forward_f64, scalar_eq_forward_f64, ==)
CMP_OP(double, ne_forward_f64, scalar_ne_forward_f64, !=)
CMP_OP(double, gt_forward_f64, scalar_gt_forward_f64, >)
CMP_OP(double, ge_forward_f64, scalar_ge_forward_f64, >=)
CMP_OP(double, lt_forward_f64, scalar_lt_forward_f64, <)
CMP_OP(double, le_forward_f64, scalar_le_forward_f64, <=)
