#include "unary_op_macros.cuh"

template<typename F>
struct ClampKernelOp {
    F min;
    F max;
};

UNARY_OP(__half, clamp_fwd_f16, clamp_bwd_f16, ClampKernelOp<__half>,
    maxg(ming(x, op.max), op.min),
    x <= op.max && x >= op.min ? 1.0 : 0.0)

UNARY_OP(float, clamp_fwd_f32, clamp_bwd_f32, ClampKernelOp<float>,
        maxg(ming(x, op.max), op.min),
        x <= op.max && x >= op.min ? 1.0 : 0.0)

UNARY_OP(double, clamp_fwd_f64, clamp_bwd_f64, ClampKernelOp<double>,
    maxg(ming(x, op.max), op.min),
    x <= op.max && x >= op.min ? 1.0 : 0.0)
    