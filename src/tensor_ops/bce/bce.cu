struct BCEKernelOp {};

extern "C" __global__ void bce_forward(
    const BCEKernelOp op,
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    float logit = lhs[i];
    float prob = rhs[i];

    out[i] = fmaxf(logit, 0.0) - logit * prob + logf(1.0 + expf(-fabsf(logit)));
}

extern "C" __global__ void bce_backward(
    const BCEKernelOp op,
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

    auto logit = lhs[i];
    auto prob = rhs[i];
    auto go = grad_out[i];

    float dfdx = 1.0 - prob - 1 / (1.0 + expf(logit));
    grad_lhs[i] += dfdx * go;

    float dfdy = -logit;
    grad_rhs[i] += dfdy * go;
}
