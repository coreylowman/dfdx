#include "cuda_utils.cuh"

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
    T one = 1.0;

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    s_avg += (one - cfg.alpha) * (g * g - s_avg);

    T avg;

    if (cfg.centered) {
        // ga = a * ga + (1 - a) * g
        g_avg += (one - cfg.alpha) * (g - g_avg);
        avg = sqrtg(s_avg - g_avg * g_avg + cfg.eps);
    } else {
        avg = sqrtg(s_avg + cfg.eps);
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

__device__ void rmsprop_update(
    const RMSpropConfig<__half> cfg,
    const size_t numel,
    __half* param,
    __half* momentum,
    __half* square_avg,
    __half* grad_avg,
    const __half* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    RMSpropConfig<float> cfg_f32 = RMSpropConfig<float> {
        cfg.lr,
        cfg.alpha,
        cfg.eps,
        cfg.centered,
        cfg.has_momentum,
        cfg.momentum,
        cfg.weight_decay_type,
        cfg.weight_decay,
    };

    float p = param[i];
    float g = grad[i];
    float s_avg = square_avg[i];
    float g_avg = grad_avg[i];
    float m = momentum[i];
    float one = 1.0;

    if (cfg_f32.weight_decay_type == L2) {
        g += cfg_f32.weight_decay * p;
    }

    s_avg += (one - cfg_f32.alpha) * (g * g - s_avg);

    float avg;

    if (cfg_f32.centered) {
        // ga = a * ga + (1 - a) * g
        g_avg += (one - cfg_f32.alpha) * (g - g_avg);
        avg = sqrtg(s_avg - g_avg * g_avg + cfg_f32.eps);
    } else {
        avg = sqrtg(s_avg + cfg_f32.eps);
    };

    g /= avg;

    if (cfg_f32.has_momentum) {
        m = m * cfg_f32.momentum + g;
        g = m * cfg_f32.lr;
    } else {
        g *= cfg_f32.lr;
    }

    if (cfg_f32.weight_decay_type == Decoupled) {
        g += cfg_f32.weight_decay * cfg_f32.lr * p;
    }

    square_avg[i] = __float2half(s_avg);
    grad_avg[i] = __float2half(g_avg);
    momentum[i] = __float2half(m);
    param[i] -= __float2half(g);
}

#define RMSPROP(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const RMSpropConfig<TYPENAME> cfg, \
    const size_t numel, \
    TYPENAME* param, \
    TYPENAME* momentum, \
    TYPENAME* square_avg, \
    TYPENAME* grad_avg, \
    const TYPENAME* grad \
) { \
    rmsprop_update(cfg, numel, param, momentum, square_avg, grad_avg, grad); \
}

RMSPROP(__half, rmsprop_update_f16);
RMSPROP(float, rmsprop_update_f32);
RMSPROP(double, rmsprop_update_f64);
