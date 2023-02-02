#include <cstdint>
#include "cuda_utils.cuh"

#define BOOLEAN_OP(NAME, OP) \
extern "C" __global__ void NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const bool *lhs, \
    const size_t *lhs_strides, \
    const bool *rhs, \
    const size_t *rhs_strides, \
    bool *out \
) { \
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (out_i >= numel) { \
        return; \
    } \
\
    unsigned int lhs_i = get_strided_index(out_i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = get_strided_index(out_i, num_dims, dims, rhs_strides); \
\
    out[out_i] = (bool)(lhs[lhs_i]) OP (bool)(rhs[rhs_i]); \
}

extern "C" __global__ void boolean_not(
    const size_t numel,
    const bool *inp,
    bool *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = !(bool)(inp[i]);
}

BOOLEAN_OP(boolean_and, &&);
BOOLEAN_OP(boolean_or, ||);
BOOLEAN_OP(boolean_xor, ^);
