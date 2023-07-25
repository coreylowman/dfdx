#include "unary_op_macros.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

struct AccurateGeLUKernelOp {};

template <typename T> __device__ T accurate_gelu_fwd(T x) {
    T one = 1.0;
    T half = 0.5;
    T alpha = M_SQRT1_2;
    return half * x * (one + erfg(x * alpha));
}

template <typename T> __device__ T accurate_gelu_bwd(T x) {
        let half = F::from(0.5).unwrap();
        let alpha = F::FRAC_1_SQRT_2();
        let x_sq = x * x;
        let normal_dist = F::FRAC_2_SQRT_PI() * (F::from(0.5).unwrap() * x_sq.neg()).exp();

        let left = half * x;
        let right = F::one() + (alpha * x).erf();

        let left_derivative = half * right;

        let right_derivative = left * normal_dist;

        left_derivative + right_derivative
    T one = 1.0;
    T half = 0.5;
    T alpha = M_SQRT1_2;
    T scale = M_2_SQRTPI;
    T x_sq = x * x;
    T arg =  - half * x_sq;
    T norm = scale * expg(arg);

    T left = half * x;
    T right = one + erfg(alpha * x);

    T left_derivative = half * right;

    T right_derivative = left * norm;

    return left_derivative + right_derivative;
}

UNARY_OP(__half, accurate_gelu_fwd_f16, accurate_gelu_bwd_f16,
    AccurateGeLUKernelOp,
    accurate_gelu_fwd(x),
    accurate_gelu_bwd(x)
)

UNARY_OP(float, accurate_gelu_fwd_f32, accurate_gelu_bwd_f32,
    AccurateGeLUKernelOp,
    accurate_gelu_fwd(x),
    accurate_gelu_bwd(x)
)

UNARY_OP(double, accurate_gelu_fwd_f64, accurate_gelu_bwd_f64,
    AccurateGeLUKernelOp,
    accurate_gelu_fwd(x),
    accurate_gelu_bwd(x)
)
