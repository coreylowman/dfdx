#include "cuda_utils.cuh"

enum WeightDecayType {
    None,
    L2,
    Decoupled
};

struct AdamConfig {
    double lr;
    double beta1;
    double beta2;
    double eps;
    WeightDecayType weight_decay_type;
    double weight_decay;
};

template<typename T>
__device__ void adam_update(
    const AdamConfig cfg,
    const size_t numel,
    const int t_int,
    T* param,
    T* moment1,
    T* moment2,
    const T* grad
) {
    T beta1 = cfg.beta1;
    T beta2 = cfg.beta2;
    T lr = cfg.lr;
    T weight_decay = cfg.weight_decay;
    T eps = cfg.eps;
    T one = 1.0;
    T t = t_int;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        T p = param[i];
        T g = grad[i];
        T m = moment1[i];
        T v = moment2[i];
    
        if (cfg.weight_decay_type == L2) {
            g += weight_decay * p;
        }
    
        m = m * beta1 + g * (one - beta1);
        v = v * beta2 + g * g * (one - beta2);
        T m_hat = m * one / (one - powg(beta1, t));
        T v_hat = v * one / (one - powg(beta2, t));
        g = lr * m_hat / (sqrtg(v_hat) + eps);
    
        if (cfg.weight_decay_type == Decoupled) {
            g += (weight_decay * lr) * p;
        }
    
        moment1[i] = m;
        moment2[i] = v;
        param[i] -= g;
    }
}

#define ADAM(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const AdamConfig cfg, \
    const size_t numel, \
    const int t, \
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

extern "C" __global__ void adam_update_amp_f16(
    const AdamConfig cfg,
    const size_t numel,
    const int t_int,
    __half* param,
    __half* moment1,
    __half* moment2,
    const __half* grad
) {
    float beta1 = cfg.beta1;
    float beta2 = cfg.beta2;
    float lr = cfg.lr;
    float weight_decay = cfg.weight_decay;
    float eps = cfg.eps;
    float one = 1.0;
    float t = t_int;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float p = param[i];
        float g = grad[i];
        float m = moment1[i];
        float v = moment2[i];
    
        if (cfg.weight_decay_type == L2) {
            g += weight_decay * p;
        }
    
        m = m * beta1 + g * (one - beta1);
        v = v * beta2 + g * g * (one - beta2);
        float m_hat = m * one / (one - powg(beta1, t));
        float v_hat = v * one / (one - powg(beta2, t));
        g = lr * m_hat / (sqrtg(v_hat) + eps);
    
        if (cfg.weight_decay_type == Decoupled) {
            g += (weight_decay * lr) * p;
        }
    
        moment1[i] = m;
        moment2[i] = v;
        param[i] -= g;
    }
}