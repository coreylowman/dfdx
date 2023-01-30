#include "unary_op_macros.cuh"

struct AbsKernelOp {};

UNARY_OP(float, abs_forward_f32, abs_backward_f32, AbsKernelOp,
        fabsf(x),
        x == 0.0 ? 0.0 : copysignf(1.0, x));

UNARY_OP(double, abs_forward_f64, abs_backward_f64, AbsKernelOp,
        fabs(x),
        x == 0.0 ? 0.0 : copysign(1.0, x));
