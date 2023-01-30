extern "C" __global__ void dropout_forward_f32(
    const float prob,
    const size_t numel,
    const float *inp,
    const float *noise,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    float scalar = (noise[i] < prob) ? 0.0 : (1.0 / (1.0 - prob));
    out[i] = inp[i] * scalar;
}

extern "C" __global__ void dropout_backward_f32(
    const float prob,
    const size_t numel,
    const float *noise,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    grad_inp[i] += (noise[i] < prob) ? 0.0 : (grad_out[i] / (1.0 - prob));
}
