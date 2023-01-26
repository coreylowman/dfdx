enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};


struct RMSpropConfig {
    float lr;
    float alpha;
    float eps;
    bool centered;
    bool has_momentum;
    float momentum;
    WeightDecayType weight_decay_type;
    float weight_decay;
};

extern "C" __global__ void rmsprop_update(
    const RMSpropConfig cfg,
    const size_t numel,
    float* param,
    float* momentum,
    float* square_avg,
    float* grad_avg,
    const float* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    float p = param[i];
    float g = grad[i];
    float s_avg = square_avg[i];
    float g_avg = grad_avg[i];
    float m = momentum[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    s_avg += (1.0 - cfg.alpha) * (g * g - s_avg);

    float avg;

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
