#include "unary_op_macros.cuh"

struct CosKernelOp {};

UNARY_OP(__half, cos_fwd_f16, cos_bwd_f16, CosKernelOp,
        cosf(x),
        -sinf(x))

UNARY_OP(float, cos_fwd_f32, cos_bwd_f32, CosKernelOp,
        cosf(x),
        -sinf(x))

UNARY_OP(double, cos_fwd_f64, cos_bwd_f64, CosKernelOp,
        cos(x),
        -sin(x))
