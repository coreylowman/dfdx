#include "unary_op_macros.cuh"

struct ReLUKernelOp {};

template<typename T>
__device__ __forceinline__ T relu_fwd(T x) {
    T zero = 0.0;
    return maxg(x, zero);
}

template<typename T>
__device__ __forceinline__ T relu_bwd(T x) {
    T zero = 0.0;
    T one = 1.0;
    return x > zero ? one : zero;
}

UNARY_OP(__half, relu_fwd_f16, relu_bwd_f16, ReLUKernelOp,
        relu_fwd(x),
        relu_bwd(x))

UNARY_OP(float, relu_fwd_f32, relu_bwd_f32, ReLUKernelOp,
        relu_fwd(x),
        relu_bwd(x))

UNARY_OP(double, relu_fwd_f64, relu_bwd_f64, ReLUKernelOp,
        relu_fwd(x),
        relu_bwd(x))
