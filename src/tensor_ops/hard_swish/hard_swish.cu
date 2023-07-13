#include "unary_op_macros.cuh"

struct HardSwishKernelOp {};

template<typename T>
__device__ __forceinline__ T hard_swish_fwd(T x) {
    T zero = 0.0;
    T three = 3.0;
    T six = 6.0;
    return x * ming(maxg(x + three, zero), six) / six;
}

template<typename T>
__device__ __forceinline__ T hard_swish_bwd(T x) {
    T minus_three = -3.0;
    T zero = 0.0;
    T three = 3.0;
    T six = 6.0;
    return x > minus_three ? ((x + x + three) / six) : zero;
}

UNARY_OP(__half, hard_swish_fwd_f16, hard_swish_bwd_f16, HardSwishKernelOp,
        hard_swish_fwd(x),
        hard_swish_bwd(x))

UNARY_OP(float, hard_swish_fwd_f32, hard_swish_bwd_f32, HardSwishKernelOp,
        hard_swish_fwd(x),
        hard_swish_bwd(x))

UNARY_OP(double, hard_swish_fwd_f64, hard_swish_bwd_f64, HardSwishKernelOp,
        hard_swish_fwd(x),
        hard_swish_bwd(x))
