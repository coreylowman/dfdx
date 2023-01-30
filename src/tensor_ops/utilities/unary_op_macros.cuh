#define LONG_UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVE) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    TYPENAME x = inp[i]; \
    FUNC \
} \
\
extern "C" __global__ void BACKWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    \
    TYPENAME x = inp[i]; \
    TYPENAME dx; \
    DERIVATIVE \
    grad_inp[i] += dx * grad_out[i]; \
}

#define UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVE) \
    LONG_UNARY_OP(TYPENAME, FORWARD, BACKWARD, OP_STRUCT, out[i] = (FUNC);, dx = (DERIVATIVE);)
