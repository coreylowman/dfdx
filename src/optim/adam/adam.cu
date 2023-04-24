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
    T one = 1.0;

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    m = m * cfg.beta1 + g * (one - cfg.beta1);
    v = v * cfg.beta2 + g * g * (one - cfg.beta2);
    T m_hat = m * one / (one - powg(cfg.beta1, t));
    T v_hat = v * one / (one - powg(cfg.beta2, t));
    g = cfg.lr * m_hat / (sqrtg(v_hat) + cfg.eps);

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    moment1[i] = m;
    moment2[i] = v;
    param[i] -= g;
}

__device__ void adam_update(
    const AdamConfig<__half> cfg,
    const size_t numel,
    const __half t,
    __half* param,
    __half* moment1,
    __half* moment2,
    const __half* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const float t_f32 = t;

    const AdamConfig<float> cfg_f32 = AdamConfig<float> {
        .lr = cfg.lr,
        .beta1 = cfg.beta1,
        .beta2 = cfg.beta2,
        .eps = cfg.eps,
        .weight_decay_type = cfg.weight_decay_type,
        .weight_decay = cfg.weight_decay,
    };

    float p = param[i];
    float g = grad[i];
    float m = moment1[i];
    float v = moment2[i];
    float one = 1.0;

    if (cfg_f32.weight_decay_type == L2) {
        g += cfg_f32.weight_decay * p;
    }

    m = m * cfg_f32.beta1 + g * (one - cfg_f32.beta1);
    v = v * cfg_f32.beta2 + g * g * (one - cfg_f32.beta2);
    float m_hat = m * one / (one - powg(cfg_f32.beta1, t_f32));
    float v_hat = v * one / (one - powg(cfg_f32.beta2, t_f32));
    g = cfg_f32.lr * m_hat / (sqrtg(v_hat) + cfg_f32.eps);

    if (cfg_f32.weight_decay_type == Decoupled) {
        g += cfg_f32.weight_decay * cfg_f32.lr * p;
    }

    moment1[i] = __float2half(m);
    moment2[i] = __float2half(v);
    param[i] -= __float2half(g);
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
