struct SinKernelOp {};

extern "C" __global__ void sin_forward(
    const SinKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = sinf(inp[i]);
}

extern "C" __global__ void sin_backward(
    const SinKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float dx = cosf(inp[i]);
    grad_inp[i] += dx * grad_out[i];
}
