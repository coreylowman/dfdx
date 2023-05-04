#include "cuda_fp16.h"

#define DROPOUT(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const TYPENAME prob, \
    const size_t numel, \
    const TYPENAME *inp, \
    const bool *mask, \
    TYPENAME *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    TYPENAME zero = 0.0; \
    TYPENAME one = 1.0; \
    TYPENAME scalar = mask[i] ? zero : (one / (one - prob)); \
    out[i] = inp[i] * scalar; \
} \
extern "C" __global__ void BWD( \
    const TYPENAME prob, \
    const size_t numel, \
    const bool *mask, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    TYPENAME zero = 0.0; \
    TYPENAME one = 1.0; \
    grad_inp[i] += mask[i] ? zero : (grad_out[i] / (one - prob)); \
}

DROPOUT(__half, dropout_fwd_f16, dropout_bwd_f16);
DROPOUT(float, dropout_fwd_f32, dropout_bwd_f32);
DROPOUT(double, dropout_fwd_f64, dropout_bwd_f64);
