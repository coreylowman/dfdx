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
