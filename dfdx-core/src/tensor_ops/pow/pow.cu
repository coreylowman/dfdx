#include "unary_op_macros.cuh"

template<typename F>
struct PowFKernelOp {
    F rhs;
};

template<typename T>
__device__ T pow_bwd(PowFKernelOp<T> op, T x) {
    T one = 1.0;
    return op.rhs * powg(x, op.rhs - one);
}

UNARY_OP(__half, pow_fwd_f16, pow_bwd_f16, PowFKernelOp<__half>,
    powg(x, op.rhs),
    pow_bwd(op, x))

UNARY_OP(float, pow_fwd_f32, pow_bwd_f32, PowFKernelOp<float>,
    powg(x, op.rhs),
    pow_bwd(op, x))

UNARY_OP(double, pow_fwd_f64, pow_bwd_f64, PowFKernelOp<double>,
    powg(x, op.rhs),
    pow_bwd(op, x))
