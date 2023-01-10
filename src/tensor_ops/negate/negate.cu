struct NegateKernelOp {};

extern "C" __global__ void negate_forward(
    const NegateKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = -inp[i];
}

extern "C" __global__ void negate_backward(
    const NegateKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = -1.0;
    grad_inp[i] += dx * grad_out[i];
}
