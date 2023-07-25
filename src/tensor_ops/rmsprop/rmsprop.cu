#include "cuda_utils.cuh"

enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};

struct RMSpropConfig {
    double lr;
    double alpha;
    double eps;
    bool centered;
    bool has_momentum;
    double momentum;
    WeightDecayType weight_decay_type;
    double weight_decay;
};

template<typename T>
__device__ void rmsprop_update(
    const RMSpropConfig cfg,
    const size_t numel,
    T* param,
    T* momentum,
    T* square_avg,
    T* grad_avg,
    const T* grad
) {
    T lr = cfg.lr;
    T alpha = cfg.alpha;
    T eps = cfg.eps;
    T momentum_ = cfg.momentum;
    T weight_decay = cfg.weight_decay;
    T one = 1.0;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        T p = param[i];
        T g = grad[i];
        T s_avg = square_avg[i];
        T g_avg = grad_avg[i];
        T m = momentum[i];
        
    
        if (cfg.weight_decay_type == L2) {
            g += weight_decay * p;
        }
    
        s_avg += (one - alpha) * (g * g - s_avg);
    
        T avg;
    
        if (cfg.centered) {
            // ga = a * ga + (1 - a) * g
            g_avg += (one - alpha) * (g - g_avg);
            avg = sqrtg(s_avg - g_avg * g_avg + eps);
        } else {
            avg = sqrtg(s_avg + eps);
        };
    
        g /= avg;
    
        if (cfg.has_momentum) {
            m = m * momentum_ + g;
            g = m * lr;
        } else {
            g *= lr;
        }
    
        if (cfg.weight_decay_type == Decoupled) {
            g += weight_decay * lr * p;
        }
    
        square_avg[i] = s_avg;
        grad_avg[i] = g_avg;
        momentum[i] = m;
        param[i] -= g;
    }
}

#define RMSPROP(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const RMSpropConfig cfg, \
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
