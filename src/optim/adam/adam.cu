#include "cuda_utils.cuh"

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
    T m_hat = m * 1.0 / (1.0 - powg(cfg.beta1, t));
    T v_hat = v * 1.0 / (1.0 - powg(cfg.beta2, t));
    g = cfg.lr * m_hat / (sqrtg(v_hat) + cfg.eps);

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    moment1[i] = m;
    moment2[i] = v;
    param[i] -= g;
}

#define ADAM(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const AdamConfig<TYPENAME> cfg, \
    const size_t numel, \
    const TYPENAME t, \
    TYPENAME* param, \
    TYPENAME* moment1, \
    TYPENAME* moment2, \
    const TYPENAME* grad \
) { \
    adam_update(cfg, numel, t, param, moment1, moment2, grad); \
}

ADAM(__half, adam_update_f16);
ADAM(float, adam_update_f32);
ADAM(double, adam_update_f64);
