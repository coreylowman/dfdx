#define DROPOUT(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const TYPENAME prob, \
    const size_t numel, \
    const TYPENAME *inp, \
    const TYPENAME *noise, \
    TYPENAME *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    auto scalar = (noise[i] < prob) ? 0.0 : (1.0 / (1.0 - prob)); \
    out[i] = inp[i] * scalar; \
} \
extern "C" __global__ void BWD( \
    const TYPENAME prob, \
    const size_t numel, \
    const TYPENAME *noise, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    grad_inp[i] += (noise[i] < prob) ? 0.0 : (grad_out[i] / (1.0 - prob)); \
}

DROPOUT(float, dropout_forward_f32, dropout_backward_f32);
DROPOUT(double, dropout_forward_f64, dropout_backward_f64);
