struct ClampKernelOp {
    float min;
    float max;
};

extern "C" __global__ void clamp_forward(
    const ClampKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = fmaxf(fminf(inp[i], op.max), op.min);
}

extern "C" __global__ void clamp_backward(
    const ClampKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = inp[i] <= op.max && inp[i] >= op.min ? 1.0 : 0.0;
    grad_inp[i] += dx * grad_out[i];
}
