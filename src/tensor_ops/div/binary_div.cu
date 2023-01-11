struct BinaryDivOp {};

extern "C" __global__ void binary_div_forward(
    const BinaryDivOp op,
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    out[i] = lhs[i] / rhs[i];
}

extern "C" __global__ void binary_div_backward(
    const BinaryDivOp op,
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

    float dfdx = 1.0 / y;
    grad_lhs[i] += dfdx * go;

    float dfdy = -x / (y * y);
    grad_rhs[i] += dfdy * go;
}
