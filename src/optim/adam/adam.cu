enum WeightDecayType {
    None,
    L2,
    Decoupled
};

template<typename T>
struct AdamConfig {
    T lr;
    T beta1;
    T beta2;
    T eps;
    WeightDecayType weight_decay_type;
    T weight_decay;
};

template<typename T>
__device__ void adam_update(
    const AdamConfig<T> cfg,
    const size_t numel,
    const T t,
    T* param,
    T* moment1,
    T* moment2,
    const T* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    T p = param[i];
    T g = grad[i];
    T m = moment1[i];
    T v = moment2[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    m = m * cfg.beta1 + g * (1.0 - cfg.beta1);
    v = v * cfg.beta2 + g * g * (1.0 - cfg.beta2);
    T m_hat = m * 1.0 / (1.0 - powf(cfg.beta1, t));
    T v_hat = v * 1.0 / (1.0 - powf(cfg.beta2, t));
    g = cfg.lr * m_hat / (sqrtf(v_hat) + cfg.eps);

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    moment1[i] = m;
    moment2[i] = v;
    param[i] -= g;
}

extern "C" __global__ void adam_update_f32(
    const AdamConfig<float> cfg,
    const size_t numel,
    const float t,
    float* param,
    float* moment1,
    float* moment2,
    const float* grad
) {
    adam_update(cfg, numel, t, param, moment1, moment2, grad);
}

extern "C" __global__ void adam_update_f64(
    const AdamConfig<double> cfg,
    const size_t numel,
    const double t,
    double* param,
    double* moment1,
    double* moment2,
    const double* grad
) {
    adam_update(cfg, numel, t, param, moment1, moment2, grad);
}