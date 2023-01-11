struct BinaryMulKernalOp {};

extern "C" __global__ void binary_mul_forward(
    const BinaryMulKernalOp op,
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    out[i] = lhs[i] * rhs[i];
}

extern "C" __global__ void binary_mul_backward(
    const BinaryMulKernalOp op,
    const size_t numel,
    const float *lhs,
    float *grad_lhs,
    const float *rhs,
    float *grad_rhs,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    auto x = lhs[i];
    auto y = rhs[i];
    auto go = grad_out[i];

    grad_lhs[i] += y * go;
    grad_rhs[i] += x * go;
}
