struct ScalarSubKernelOp {
    float scalar;
};

extern "C" __global__ void scalar_sub_forward(
    const ScalarSubKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] = inp[i] - op.scalar;
}

extern "C" __global__ void scalar_sub_backward(
    const ScalarSubKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float df = 1.0;
    grad_inp[i] += df * grad_out[i];
}
