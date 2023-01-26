struct GeLUKernelOp {};

extern "C" __global__ void gelu_forward(
    const GeLUKernelOp op,
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    constexpr float fastCoeff = 0.044715;
    float x = inp[i];
    float x_sq = x * x;
    float x_cube = x_sq * x;

    float alpha = x + fastCoeff * x_cube;

    float y = 0.5 * x * (1.0 + tanh(M_2_SQRTPI * M_SQRT1_2 * alpha));
    out[i] = y;
}

extern "C" __global__ void gelu_backward(
    const GeLUKernelOp op,
    const size_t numel,
    const float *inp,
    float *grad_inp,
    const float *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float kBeta = M_2_SQRTPI * M_SQRT2 * 0.5;                       
    constexpr float fastCoeff = 0.044715;
    float x = inp[i];
    float x_sq = x * x;
    float x_cube = x_sq * x;
    float inner = kBeta * (x + fastCoeff * x_cube);
    float tanh_inner = tanh(inner);

    float left = 0.5 * x;
    float right = 1.0 + tanh_inner;
    
    float left_derivative = 0.5 * right;

    float tanh_derivative = 1.0 - tanh_inner * tanh_inner;
    float inner_derivative = kBeta * (1.0 + 3.0 * fastCoeff * x_sq);
    float right_derivative = left * tanh_derivative * inner_derivative;
    float dx = left_derivative + right_derivative;
    grad_inp[i] += dx * grad_out[i];
}
