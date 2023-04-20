#include "cuda_utils.cuh"

template<typename T>
__device__ void choose_fwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const bool *cond,
    const size_t *cond_strides,
    const T *lhs,
    const size_t *lhs_strides,
    const T *rhs,
    const size_t *rhs_strides,
    T *out
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

template<typename T>
__device__ void choose_bwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const bool *cond,
    const size_t *cond_strides,
    T *grad_lhs,
    const size_t *lhs_strides,
    T *grad_rhs,
    const size_t *rhs_strides,
    const T *grad_out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(out_i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(out_i, num_dims, dims, rhs_strides);
    unsigned int cond_i = get_strided_index(out_i, num_dims, dims, cond_strides);

    auto go = grad_out[out_i];
    T* out_loc = cond[cond_i] ? grad_lhs + lhs_i : grad_rhs + rhs_i;

    atomicAdd(out_loc, go);
}

#define CHOOSE(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const bool *cond, \
    const size_t *cond_strides, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
    const size_t *rhs_strides, \
    TYPENAME *out \
) { \
    choose_fwd(numel, num_dims, dims, cond, cond_strides, lhs, lhs_strides, rhs, rhs_strides, out); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const bool *cond, \
    const size_t *cond_strides, \
    TYPENAME *grad_lhs, \
    const size_t *lhs_strides, \
    TYPENAME *grad_rhs, \
    const size_t *rhs_strides, \
    const TYPENAME *grad_out \
) { \
    choose_bwd(numel, num_dims, dims, cond, cond_strides, grad_lhs, lhs_strides, grad_rhs, rhs_strides, grad_out); \
}

CHOOSE(__half, choose_fwd_f16, choose_bwd_f16);
CHOOSE(float, choose_fwd_f32, choose_bwd_f32);
CHOOSE(double, choose_fwd_f64, choose_bwd_f64);
