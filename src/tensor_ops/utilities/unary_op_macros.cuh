#define UNARY_OP(FORWARD, BACKWARD, OP_STRUCT, FUNC, DERIVATIVE) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const float *inp, \
    float *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    float x = inp[i]; \
    out[i] = (FUNC); \
} \
 \
extern "C" __global__ void BACKWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const float *inp, \
    float *grad_inp, \
    const float *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    \
    float x = inp[i]; \
    float dx = (DERIVATIVE); \
    grad_inp[i] += dx * grad_out[i]; \
}
