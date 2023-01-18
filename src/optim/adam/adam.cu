enum WeightDecayType {
    None,
    L2,
    Decoupled
};

struct AdamConfig {
    float lr;
    float beta1;
    float beta2;
    float eps;
    WeightDecayType weight_decay_type;
    float weight_decay;
};

extern "C" __global__ void adam_update(
    const AdamConfig cfg,
    const size_t numel,
    const float t,
    float* param,
    float* moment1,
    float* moment2,
    const float* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    float p = param[i];
    float g = grad[i];
    float m = moment1[i];
    float v = moment2[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    m = m * cfg.beta1 + g * (1.0 - cfg.beta1);
    v = v * cfg.beta2 + g * g * (1.0 - cfg.beta2);
    float m_hat = m * 1.0 / (1.0 - powf(cfg.beta1, t));
    float v_hat = v * 1.0 / (1.0 - powf(cfg.beta2, t));
    g = cfg.lr * m_hat / (sqrtf(v_hat) + cfg.eps);

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    moment1[i] = m;
    moment2[i] = v;
    param[i] -= g;
}
