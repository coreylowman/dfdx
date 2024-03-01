#include "unary_op_macros.cuh"

struct SiLUKernelOp {};

// x / (1 + e^-x)
template<typename T>
__device__ __forceinline__ T silu_fwd(T x) {
    T one = 1.0;
    return x / (one + expg(-x));
}

// (1 + e^-x + x * e^-x) / (1 + e^-x)^2
// alternative: (e^x (1 + e^x + x)) / (1 + e^x)^2
template<typename T>
__device__ __forceinline__ T silu_bwd(T x) {
    T one = 1.0;
    T exp_nx = expg(-x);
    T denom_sqrt = (one + exp_nx);
    return (one + exp_nx + x * exp_nx) / (denom_sqrt * denom_sqrt);
}

UNARY_OP(__half, silu_fwd_f16, silu_bwd_f16, SiLUKernelOp,
        silu_fwd(x),
        silu_bwd(x))

UNARY_OP(float, silu_fwd_f32, silu_bwd_f32, SiLUKernelOp,
        silu_fwd(x),
        silu_bwd(x))

UNARY_OP(double, silu_fwd_f64, silu_bwd_f64, SiLUKernelOp,
        silu_fwd(x),
        silu_bwd(x))
