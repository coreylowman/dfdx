#include "unary_op_macros.cuh"

struct SqrtKernelOp {};

UNARY_OP(float, sqrt_fwd_f32, sqrt_bwd_f32, SqrtKernelOp,
        sqrtf(x),
        1 / (y + y))

UNARY_OP(double, sqrt_fwd_f64, sqrt_bwd_f64, SqrtKernelOp,
        sqrt(x),
        1 / (y + y))
        