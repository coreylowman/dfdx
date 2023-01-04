struct ExpKernelOp {};

extern "C" __global__ void exp_forward(
    const ExpKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = exp(inp[i]);
}

extern "C" __global__ void exp_backward(
    const ExpKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    grad_inp[i] += exp(inp[i]) * grad_out[i];
}