struct SquareKernelOp {};

extern "C" __global__ void square_forward(
    const SquareKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = inp[i] * inp[i];
}

extern "C" __global__ void square_backward(
    const SquareKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = 2.0 * inp[i];
    grad_inp[i] += dx * grad_out[i];
}
