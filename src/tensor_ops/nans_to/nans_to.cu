struct NansToKernelOp {
    float x;
};

extern "C" __global__ void nans_to_forward(
    const NansToKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = isnan(inp[i]) ? op.x : inp[i];
}

extern "C" __global__ void nans_to_backward(
    const NansToKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = isnan(inp[i]) ? 0.0 : 1.0;
    grad_inp[i] += dx * grad_out[i];
}
