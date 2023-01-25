struct PowFKernelOp {
    float rhs;
};

extern "C" __global__ void pow_forward(
    const PowFKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = powf(inp[i], op.rhs);
}

extern "C" __global__ void pow_backward(
    const PowFKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = op.rhs * powf(inp[i], op.rhs - 1.0);
    grad_inp[i] += dx * grad_out[i];
}
