#include "unary_op_macros.cuh"

struct SigmoidKernelOp {};

template<typename T>
__device__ __forceinline__ T sigmoid_fwd(T x) {
    T one = 1.0;
    return one / (one + expg(-x));
}

template<typename T>
__device__ __forceinline__ T sigmoid_bwd(T y) {
    T one = 1.0;
    return y * (one - y);
}

UNARY_OP(__half, sigmoid_fwd_f16, sigmoid_bwd_f16, SigmoidKernelOp,
        sigmoid_fwd(x),
        sigmoid_bwd(y))

UNARY_OP(float, sigmoid_fwd_f32, sigmoid_bwd_f32, SigmoidKernelOp,
        sigmoid_fwd(x),
        sigmoid_bwd(y))

UNARY_OP(double, sigmoid_fwd_f64, sigmoid_bwd_f64, SigmoidKernelOp,
        sigmoid_fwd(x),
        sigmoid_bwd(y))
        