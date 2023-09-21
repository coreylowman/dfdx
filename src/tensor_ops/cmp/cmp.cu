#include "cuda_utils.cuh"

#define CMP_OP(TYPENAME, FWD, SCALAR_FWD, SYMBOL) \
extern "C" __global__ void FWD( \
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
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
        unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
        unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
        out[out_i] = lhs[lhs_i] SYMBOL rhs[rhs_i]; \
    } \
} \
\
extern "C" __global__ void SCALAR_FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    TYPENAME scalar, \
    bool *out, \
    const size_t *out_strides \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
        unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
        out[out_i] = lhs[lhs_i] SYMBOL scalar; \
    } \
}

CMP_OP(__half, eq_fwd_f16, scalar_eq_fwd_f16, ==)
CMP_OP(__half, ne_fwd_f16, scalar_ne_fwd_f16, !=)
CMP_OP(__half, gt_fwd_f16, scalar_gt_fwd_f16, >)
CMP_OP(__half, ge_fwd_f16, scalar_ge_fwd_f16, >=)
CMP_OP(__half, lt_fwd_f16, scalar_lt_fwd_f16, <)
CMP_OP(__half, le_fwd_f16, scalar_le_fwd_f16, <=)

CMP_OP(float, eq_fwd_f32, scalar_eq_fwd_f32, ==)
CMP_OP(float, ne_fwd_f32, scalar_ne_fwd_f32, !=)
CMP_OP(float, gt_fwd_f32, scalar_gt_fwd_f32, >)
CMP_OP(float, ge_fwd_f32, scalar_ge_fwd_f32, >=)
CMP_OP(float, lt_fwd_f32, scalar_lt_fwd_f32, <)
CMP_OP(float, le_fwd_f32, scalar_le_fwd_f32, <=)

CMP_OP(double, eq_fwd_f64, scalar_eq_fwd_f64, ==)
CMP_OP(double, ne_fwd_f64, scalar_ne_fwd_f64, !=)
CMP_OP(double, gt_fwd_f64, scalar_gt_fwd_f64, >)
CMP_OP(double, ge_fwd_f64, scalar_ge_fwd_f64, >=)
CMP_OP(double, lt_fwd_f64, scalar_lt_fwd_f64, <)
CMP_OP(double, le_fwd_f64, scalar_le_fwd_f64, <=)
