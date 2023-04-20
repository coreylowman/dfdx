#include "cuda_fp16.h"

enum MomentumType {
    None,
    Classic,
    Nesterov,
};

enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};

template<typename T>
struct SgdConfig {
    T lr;
    MomentumType momentum_type;
    T momentum;
    WeightDecayType weight_decay_type;
    T weight_decay;
};

template<typename T>
__device__ void sgd_update(
    const SgdConfig<T> cfg,
    const size_t numel,
    T* param,
    T* velocity,
    const T* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    T p = param[i];
    T g = grad[i];
    T v = velocity[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    if (cfg.momentum_type == Classic) {
        v = g + cfg.momentum * v;
        g = v * cfg.lr;
    } else if (cfg.momentum_type == Nesterov) {
        v = g + cfg.momentum * v;
        g = (g + cfg.momentum * v) * cfg.lr;
    } else {
        g *= cfg.lr;
    }

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    velocity[i] = v;
    param[i] -= g;
}

#define SGD(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const SgdConfig<TYPENAME> cfg, \
    const size_t numel, \
    TYPENAME* param, \
    TYPENAME* velocity, \
    const TYPENAME* grad \
) { \
    sgd_update(cfg, numel, param, velocity, grad); \
}

SGD(__half, sgd_update_f16);
SGD(float, sgd_update_f32);
SGD(double, sgd_update_f64);
