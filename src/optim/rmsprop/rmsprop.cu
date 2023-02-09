enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};

template<typename T>
struct RMSpropConfig {
    T lr;
    T alpha;
    T eps;
    bool centered;
    bool has_momentum;
    T momentum;
    WeightDecayType weight_decay_type;
    T weight_decay;
};

template<typename T>
__device__ void rmsprop_update(
    const RMSpropConfig<T> cfg,
    const size_t numel,
    T* param,
    T* momentum,
    T* square_avg,
    T* grad_avg,
    const T* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    T p = param[i];
    T g = grad[i];
    T s_avg = square_avg[i];
    T g_avg = grad_avg[i];
    T m = momentum[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    s_avg += (1.0 - cfg.alpha) * (g * g - s_avg);

    T avg;

    if (cfg.centered) {
        // ga = a * ga + (1 - a) * g
        g_avg += (1.0 - cfg.alpha) * (g - g_avg);
        avg = sqrtf(s_avg - g_avg * g_avg + cfg.eps);
    } else {
        avg = sqrtf(s_avg + cfg.eps);
    };

    g /= avg;

    if (cfg.has_momentum) {
        m = m * cfg.momentum + g;
        g = m * cfg.lr;
    } else {
        g *= cfg.lr;
    }

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    square_avg[i] = s_avg;
    grad_avg[i] = g_avg;
    momentum[i] = m;
    param[i] -= g;
}

extern "C" __global__ void rmsprop_update_f32(
    const RMSpropConfig<float> cfg,
    const size_t numel,
    float* param,
    float* momentum,
    float* square_avg,
    float* grad_avg,
    const float* grad
) {
    rmsprop_update(cfg, numel, param, momentum, square_avg, grad_avg, grad);
}

extern "C" __global__ void rmsprop_update_f64(
    const RMSpropConfig<double> cfg,
    const size_t numel,
    double* param,
    double* momentum,
    double* square_avg,
    double* grad_avg,
    const double* grad
) {
    rmsprop_update(cfg, numel, param, momentum, square_avg, grad_avg, grad);
}