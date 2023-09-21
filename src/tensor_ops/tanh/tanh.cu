#include "unary_op_macros.cuh"

struct TanhKernelOp {};

template<typename T>
__device__ __forceinline__ T tanh_bwd(T y) {
    T one = 1.0;
    return one - y * y;
}

UNARY_OP(__half, tanh_fwd_f16, tanh_bwd_f16, TanhKernelOp,
        tanhg(x),
        tanh_bwd(y))

UNARY_OP(float, tanh_fwd_f32, tanh_bwd_f32, TanhKernelOp,
        tanhg(x),
        tanh_bwd(y))

UNARY_OP(double, tanh_fwd_f64, tanh_bwd_f64, TanhKernelOp,
        tanhg(x),
        tanh_bwd(y))
        