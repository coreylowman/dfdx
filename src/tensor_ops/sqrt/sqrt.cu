struct SqrtKernelOp {};

extern "C" __global__ void sqrt_forward(
    const SqrtKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = sqrt(inp[i]);
}

extern "C" __global__ void sqrt_backward(
    const SqrtKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = 0.5 / sqrt(inp[i]);
    grad_inp[i] += dx * grad_out[i];
}
