struct MaximumKernalOp {};

extern "C" __global__ void maximum_forward(
    const MaximumKernalOp op,
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    out[i] = fmaxf(lhs[i], rhs[i]);
}

extern "C" __global__ void maximum_backward(
    const MaximumKernalOp op,
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

    float dfdx, dfdy;

    if (x > y) {
        dfdx = 1.0;
        dfdy = 0.0;
    } else if (x < y) {
        dfdx = 0.0;
        dfdy = 1.0;
    } else {
        dfdx = 0.5;
        dfdy = 0.5;
    }

    grad_lhs[i] += dfdx * go;
    grad_rhs[i] += dfdy * go;
}
