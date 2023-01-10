struct SigmoidKernelOp {};

extern "C" __global__ void sigmoid_forward(
    const SigmoidKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = 1.0 / (1.0 + expf(-inp[i]));
}

extern "C" __global__ void sigmoid_backward(
    const SigmoidKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float fx = 1.0 / (1.0 + expf(-inp[i]));
    float dx = fx * (1.0 - fx);
    grad_inp[i] += dx * grad_out[i];
}
