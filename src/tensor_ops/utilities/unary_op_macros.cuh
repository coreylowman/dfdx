#include "cuda_utils.cuh"

#define LONG_UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVE) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = inp ? inp[i] : out[i]; \
        FUNC \
    } \
} \
\
extern "C" __global__ void BACKWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *out, \
    const TYPENAME *grad_out \
) { \
    TYPENAME zero = 0.0; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = inp ? inp[i] : zero; \
        TYPENAME y = out ? out[i] : zero; \
        TYPENAME dx; \
        DERIVATIVE \
        grad_inp[i] += dx * grad_out[i]; \
    } \
}

#define UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVE) \
    LONG_UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, out[i] = (FUNC);, dx = (DERIVATIVE);)
