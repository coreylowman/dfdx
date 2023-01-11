struct HuberErrorOp {
    float delta;
};

extern "C" __global__ void huber_error_forward(
    const HuberErrorOp op,
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    float a = lhs[i] - rhs[i];

    if (fabsf(a) < op.delta) {
        out[i] = a * a * 0.5;
    } else {
        out[i] = op.delta * (fabsf(a) - 0.5 * op.delta);
    }
}

extern "C" __global__ void huber_error_backward(
    const HuberErrorOp op,
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

    auto a = lhs[i] - rhs[i];
    auto go = grad_out[i];

    float dfdx, dfdy;

    if (a == 0.0) {
        dfdx = 0.0;
    } else if (fabsf(a) < op.delta) {
        dfdx = a;
    } else {
        dfdx = copysignf(op.delta, a);
    }

    dfdy = -dfdx;

    grad_lhs[i] += dfdx * go;
    grad_rhs[i] += dfdy * go;
}
