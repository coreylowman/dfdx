#include "unary_op_macros.cuh"

template<typename F>
struct ClampKernelOp {
    F min;
    F max;
};

UNARY_OP(float, clamp_fwd_f32, clamp_bwd_f32, ClampKernelOp<float>,
        fmaxf(fminf(x, op.max), op.min),
        x <= op.max && x >= op.min ? 1.0 : 0.0)

UNARY_OP(double, clamp_fwd_f64, clamp_bwd_f64, ClampKernelOp<double>,
    fmax(fmin(x, op.max), op.min),
    x <= op.max && x >= op.min ? 1.0 : 0.0)
    