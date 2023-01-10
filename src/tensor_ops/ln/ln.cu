struct LnKernelOp {};

extern "C" __global__ void ln_forward(
    const LnKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = logf(inp[i]);
}

extern "C" __global__ void ln_backward(
    const LnKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = 1.0 / inp[i];
    grad_inp[i] += dx * grad_out[i];
}
