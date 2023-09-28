#include "unary_op_macros.cuh"

struct SigmoidrKernelOp {};

template<typename T>
__device__ __forceinline__ T sigmoidr_fwd(T x) {
    T one = 1.0;
    return one / (one + expg(-x));
}

template<typename T>
__device__ __forceinline__ T sigmoidr_bwd(T y) {
    T one = 1.0;
  T d = y * (one - y);
    return max(d, 0.0000001);
}

UNARY_OP(__half, sigmoidr_fwd_f16, sigmoidr_bwd_f16, SigmoidrKernelOp,
        sigmoidr_fwd(x),
        sigmoidr_bwd(y))

UNARY_OP(float, sigmoidr_fwd_f32, sigmoidr_bwd_f32, SigmoidrKernelOp,
        sigmoidr_fwd(x),
        sigmoidr_bwd(y))

UNARY_OP(double, sigmoidr_fwd_f64, sigmoidr_bwd_f64, SigmoidrKernelOp,
        sigmoidr_fwd(x),
        sigmoidr_bwd(y))
        