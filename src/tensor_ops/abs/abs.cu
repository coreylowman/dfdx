struct AbsKernelOp {};

extern "C" __global__ void abs_forward(
    const AbsKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = abs(inp[i]);
}

extern "C" __global__ void abs_backward(
    const AbsKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = inp[i] == 0.0 ? 0.0 : (signbit(inp[i]) ? 1.0 : -1.0);
    grad_inp[i] += dx * grad_out[i];
}