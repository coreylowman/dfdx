struct ScalarDivKernelOp {
    float scalar;
};

extern "C" __global__ void scalar_div_forward(
    const ScalarDivKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = inp[i] / op.scalar;
}

extern "C" __global__ void scalar_div_backward(
    const ScalarDivKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    grad_inp[i] += grad_out[i] / op.scalar;
}