#include "cuda_utils.cuh"

enum WeightDecayType {
    None,
    L2,
    Decoupled
};

struct ProdigyConfig1 {
    const size_t numel;
    const int k_int;
    //
    double lr;
    double beta1;
    double beta2;
    double beta3;
    WeightDecayType weight_decay_type;
    double weight_decay;
    double bias_correction;
    bool safeguard_warmup;
    double d0;
};

template<typename T>
__device__ void prodigy_update1(
    const ProdigyConfig1 cfg,
    const double d,

    // temporaries for sum-reduction.
    // are written into and read back by host
    T* d_numerators,
    T* d_denoms,

    // parameter-related tensors.
    // some are overwritten
    const T* param,
    T* s,
    T* p0,
    T* p0b,
    T* moment1,
    T* moment2,
    const T* grad
) {
    const size_t numel = cfg.numel;
    const int k_int = cfg.k_int;
    T lr = cfg.lr;
    T beta1 = cfg.beta1;
    T beta2 = cfg.beta2;
    T beta3 = cfg.beta3;
    T weight_decay = cfg.weight_decay;
    T bias_correction = cfg.bias_correction;
    T d0 = cfg.d0;
    T zero = 0.0;
    T one = 1.0;
    T k = k_int;

    unsigned int cu_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cu_stride = blockDim.x * gridDim.x;

    if (cu_index >= numel) {
        return;
    }

    T d_ = d;
    T dlr = d_ * lr * bias_correction;

    // thread-local d_numerator and d_denom
    T d_numerator_ = zero;
    T d_denom = zero;
    // those values will be sum-reduced by all threads and all blocks

    for (unsigned int i = cu_index; i < numel; i += cu_stride) {
        T p_ = param[i];
        T g = grad[i];
        T s_ = s[i];
        T m_ = moment1[i];
        T v_ = moment2[i];

        // initialize p0 if needed
        if (p0b[i] == zero) {
            p0b[i] = one;
            p0[i] = p_;
        }
        T p0_ = p0[i];

        if (cfg.weight_decay_type == L2) {
            g += weight_decay * p_;
        }

        if (lr > zero) {
            d_numerator_ += (d_ / d0) * dlr * (g * (p0_ - p_));

            m_ = m_ * beta1 + d_ * g * (one - beta1);
            v_ = v_ * beta2 + d_ * d_ * g * g * (one - beta2);

            if (cfg.safeguard_warmup) {
                s_ = s_ * beta3 + g * d_ * d_ / d0;
            } else {
                s_ = s_ * beta3 + g * d_ * dlr / d0;
            }

            d_denom += absg(s_);
        }

        s[i] = s_;
        moment1[i] = m_;
        moment2[i] = v_;
    }

    // prepares the values for sum-reduction
    d_numerators[cu_index] = d_numerator_;
    d_denoms[cu_index] = d_denom;

    return;
}

#define PRODIGY1(TYPENAME, FN1) \
extern "C" __global__ void FN1( \
    const ProdigyConfig1 cfg, \
    const double d, \
    TYPENAME* d_numerators, \
    TYPENAME* d_denoms, \
    const TYPENAME* param, \
    TYPENAME* s, \
    TYPENAME* p0, \
    TYPENAME* p0b, \
    TYPENAME* moment1, \
    TYPENAME* moment2, \
    const TYPENAME* grad \
) { \
    prodigy_update1(cfg, d, d_numerators, d_denoms, param, s, p0, p0b, moment1, moment2, grad); \
}

PRODIGY1(__half, prodigy_update1_f16);
PRODIGY1(float, prodigy_update1_f32);
PRODIGY1(double, prodigy_update1_f64);



extern "C" __global__ void prodigy_update1_amp_f16(
    const ProdigyConfig1 cfg,
    const double d,

    // temporaries for sum-reduction.
    // are written into and read back by host
    __half* d_numerators,
    __half* d_denoms,

    // parameter-related tensors.
    // some are overwritten
    const __half* param,
    __half* s,
    __half* p0,
    __half* p0b,
    __half* moment1,
    __half* moment2,
    const __half* grad
) {
    const size_t numel = cfg.numel;
    const int k_int = cfg.k_int;
    float lr = cfg.lr;
    float beta1 = cfg.beta1;
    float beta2 = cfg.beta2;
    float beta3 = cfg.beta3;
    float weight_decay = cfg.weight_decay;
    float bias_correction = cfg.bias_correction;
    float d0 = cfg.d0;
    float zero = 0.0;
    __half zero_half = zero;
    float one = 1.0;
    float k = k_int;

    unsigned int cu_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cu_stride = blockDim.x * gridDim.x;

    if (cu_index >= numel) {
        return;
    }

    float d_ = d;
    float dlr = d_ * lr * bias_correction;

    // thread-local d_numerator and d_denom
    float d_numerator_ = zero;
    float d_denom = zero;
    // those values will be sum-reduced by all threads and all blocks

    for (unsigned int i = cu_index; i < numel; i += cu_stride) {
        float p_ = param[i];
        float g = grad[i];
        float s_ = s[i];
        float m_ = moment1[i];
        float v_ = moment2[i];

        // initialize p0 if needed
        if (p0b[i] == zero_half) {
            p0b[i] = one;
            p0[i] = p_;
        }
        float p0_ = p0[i];

        if (cfg.weight_decay_type == L2) {
            g += weight_decay * p_;
        }

        if (lr > zero) {
            d_numerator_ += (d_ / d0) * dlr * (g * (p0_ - p_));

            m_ = m_ * beta1 + d_ * g * (one - beta1);
            v_ = v_ * beta2 + d_ * d_ * g * g * (one - beta2);

            if (cfg.safeguard_warmup) {
                s_ = s_ * beta3 + g * d_ * d_ / d0;
            } else {
                s_ = s_ * beta3 + g * d_ * dlr / d0;
            }

            d_denom += absg(s_);
        }

        s[i] = s_;
        moment1[i] = m_;
        moment2[i] = v_;
    }

    // prepares the values for sum-reduction
    d_numerators[cu_index] = d_numerator_;
    d_denoms[cu_index] = d_denom;

    return;
}

struct ProdigyConfig2 {
    const size_t numel;
    double lr;
    double eps;
    WeightDecayType weight_decay_type;
    double weight_decay;
    double bias_correction;
};

template<typename T>
__device__ void prodigy_update2(
    const ProdigyConfig2 cfg,
    const double d_old,

    // parameter-related tensors.
    // some are overwritten
    T* param,
    const T* moment1,
    const T* moment2
) {
    const size_t numel = cfg.numel;
    T lr = cfg.lr;
    T eps = cfg.eps;
    T weight_decay = cfg.weight_decay;
    T bias_correction = cfg.bias_correction;
    T one = 1.0;

    T d_old_ = d_old;
    T dlr_old = d_old_ * lr * bias_correction;

    unsigned int cu_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cu_stride = blockDim.x * gridDim.x;

    if (cu_index >= numel) {
        return;
    }

    for (unsigned int i = cu_index; i < numel; i += cu_stride) {
        T p_ = param[i];
        T m_ = moment1[i];
        T v_ = moment2[i];

        T denom = sqrtg(v_) + d_old_ * eps;
        if (cfg.weight_decay_type == Decoupled) {
            p_ *= one - weight_decay * dlr_old;
        }

        p_ -= dlr_old * m_ / denom;

        param[i] = p_;
    }
}

#define PRODIGY2(TYPENAME, FN2) \
extern "C" __global__ void FN2( \
    const ProdigyConfig2 cfg, \
    const double d, \
    TYPENAME* param, \
    const TYPENAME* moment1, \
    const TYPENAME* moment2 \
) { \
    prodigy_update2(cfg, d, param, moment1, moment2); \
}

PRODIGY2(__half, prodigy_update2_f16);
PRODIGY2(float, prodigy_update2_f32);
PRODIGY2(double, prodigy_update2_f64);

extern "C" __global__ void prodigy_update2_amp_f16(
    const ProdigyConfig2 cfg,
    const double d_old,

    // parameter-related tensors.
    // some are overwritten
    __half* param,
    const __half* moment1,
    const __half* moment2
) {
    const size_t numel = cfg.numel;
    float lr = cfg.lr;
    float eps = cfg.eps;
    float weight_decay = cfg.weight_decay;
    float bias_correction = cfg.bias_correction;
    float one = 1.0;

    float d_old_ = d_old;
    float dlr_old = d_old_ * lr * bias_correction;

    unsigned int cu_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cu_stride = blockDim.x * gridDim.x;

    if (cu_index >= numel) {
        return;
    }

    for (unsigned int i = cu_index; i < numel; i += cu_stride) {
        float p_ = param[i];
        float m_ = moment1[i];
        float v_ = moment2[i];

        float denom = sqrtg(v_) + d_old_ * eps;
        if (cfg.weight_decay_type == Decoupled) {
            p_ *= one - weight_decay * dlr_old;
        }

        p_ -= dlr_old * m_ / denom;

        param[i] = p_;
    }
}