
extern "C" __global__ void reshape_forward(
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = inp[i];
}

extern "C" __global__ void reshape_backward(
    const size_t numel,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    grad_inp[i] += grad_out[i];
}
