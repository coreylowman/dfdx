#include "unary_op_macros.cuh"

template<typename F>
struct ClampKernelOp {
    F min;
    F max;
};

UNARY_OP(float, clamp_forward_f32, clamp_backward_f32, ClampKernelOp<float>,
        fmaxf(fminf(x, op.max), op.min),
        x <= op.max && x >= op.min ? 1.0 : 0.0)

UNARY_OP(double, clamp_forward_f64, clamp_backward_f64, ClampKernelOp<double>,
    fmax(fmin(x, op.max), op.min),
    x <= op.max && x >= op.min ? 1.0 : 0.0)
    