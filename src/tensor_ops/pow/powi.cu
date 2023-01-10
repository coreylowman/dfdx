struct PowIKernelOp {
    int rhs;
};

extern "C" __global__ void powi_forward(
    const PowIKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    // Intentionally uses the 64 bit version of pow to ensure that the exponent
    // isn't rounded
    out[i] = pow(inp[i], op.rhs);
}

extern "C" __global__ void powi_backward(
    const PowIKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = op.rhs * pow(inp[i], op.rhs - 1);
    grad_inp[i] += dx * grad_out[i];
}
