#include "unary_op_macros.cuh"

struct CosKernelOp {};

UNARY_OP(float, cos_forward_f32, cos_backward_f32, CosKernelOp,
        cosf(x),
        -sinf(x))

UNARY_OP(double, cos_forward_f64, cos_backward_f64, CosKernelOp,
        cos(x),
        -sin(x))
        