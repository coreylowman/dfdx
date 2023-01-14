struct TanhKernelOp {};

extern "C" __global__ void tanh_forward(
    const TanhKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = tanhf(inp[i]);
}

extern "C" __global__ void tanh_backward(
    const TanhKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float fx = tanhf(inp[i]);
    float dx = 1 - fx * fx;
    grad_inp[i] += dx * grad_out[i];
}
