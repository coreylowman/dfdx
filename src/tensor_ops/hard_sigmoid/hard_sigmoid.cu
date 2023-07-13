#include "unary_op_macros.cuh"

struct HardSigmoidKernelOp {};

template<typename T>
__device__ __forceinline__ T hard_sigmoid_fwd(T x) {
    T zero = 0.0;
    T three = 3.0;
    T six = 6.0;
    return ming(maxg(x + three, zero), six) / six;
}

template<typename T>
__device__ __forceinline__ T hard_sigmoid_bwd(T y) {
    T one_sixth = 1.0 / 6.0;
    T zero = 0.0;
    T one = 1.0;
    return y > zero ? y < one ? one_sixth : zero : zero;
}

UNARY_OP(__half, hard_sigmoid_fwd_f16, hard_sigmoid_bwd_f16, HardSigmoidKernelOp,
        hard_sigmoid_fwd(x),
        hard_sigmoid_bwd(y))

UNARY_OP(float, hard_sigmoid_fwd_f32, hard_sigmoid_bwd_f32, HardSigmoidKernelOp,
        hard_sigmoid_fwd(x),
        hard_sigmoid_bwd(y))

UNARY_OP(double, hard_sigmoid_fwd_f64, hard_sigmoid_bwd_f64, HardSigmoidKernelOp,
        hard_sigmoid_fwd(x),
        hard_sigmoid_bwd(y))
