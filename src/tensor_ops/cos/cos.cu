struct CosKernelOp {};

extern "C" __global__ void cos_forward(
    const CosKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = cosf(inp[i]);
}

extern "C" __global__ void cos_backward(
    const CosKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = -sinf(inp[i]);
    grad_inp[i] += dx * grad_out[i];
}
