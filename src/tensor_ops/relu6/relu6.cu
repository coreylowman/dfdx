#include "unary_op_macros.cuh"

struct ReLU6KernelOp {};

template<typename T>
__device__ __forceinline__ T relu6_fwd(T x) {
    T zero = 0.0;
    T six = 6.0;
    return ming(maxg(x, zero), six);
}

template<typename T>
__device__ __forceinline__ T relu6_bwd(T x) {
    T zero = 0.0;
    T one = 1.0;
    T six = 6.0;
    return x > zero ? x < six ? one : zero : zero;
}

UNARY_OP(__half, relu6_fwd_f16, relu6_bwd_f16, ReLU6KernelOp,
        relu6_fwd(x),
        relu6_bwd(x))

UNARY_OP(float, relu6_fwd_f32, relu6_bwd_f32, ReLU6KernelOp,
        relu6_fwd(x),
        relu6_bwd(x))

UNARY_OP(double, relu6_fwd_f64, relu6_bwd_f64, ReLU6KernelOp,
        relu6_fwd(x),
        relu6_bwd(x))
