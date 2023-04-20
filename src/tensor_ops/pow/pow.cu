#include "unary_op_macros.cuh"

template<typename F>
struct PowFKernelOp {
    F rhs;
};

UNARY_OP(__half, pow_fwd_f16, pow_bwd_f16, PowFKernelOp<__half>,
    powf(x, op.rhs),
    op.rhs * powf(x, op.rhs - 1.0))

UNARY_OP(float, pow_fwd_f32, pow_bwd_f32, PowFKernelOp<float>,
        powf(x, op.rhs),
        op.rhs * powf(x, op.rhs - 1.0))

UNARY_OP(double, pow_fwd_f64, pow_bwd_f64, PowFKernelOp<double>,
    pow(x, op.rhs),
    op.rhs * pow(x, op.rhs - 1.0))
    