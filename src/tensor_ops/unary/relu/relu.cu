struct ReLUKernelOp {};

extern "C" __global__ void relu_forward(
    const ReLUKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = fmaxf(inp[i], 0.0);
}

extern "C" __global__ void relu_backward(
    const ReLUKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = inp[i] > 0.0 ? 1.0 : 0.0;
    grad_inp[i] += dx * grad_out[i];
}
