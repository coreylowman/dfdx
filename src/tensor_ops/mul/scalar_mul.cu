struct ScalarMulKernelOp {
    float scalar;
};

extern "C" __global__ void scalar_mul_forward(
    const ScalarMulKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = inp[i] * op.scalar;
}

extern "C" __global__ void scalar_mul_backward(
    const ScalarMulKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float df = op.scalar;
    grad_inp[i] += df * grad_out[i];
}
