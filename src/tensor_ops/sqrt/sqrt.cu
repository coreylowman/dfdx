#include "unary_op_macros.cuh"

struct SqrtKernelOp {};

UNARY_OP(__half, sqrt_fwd_f16, sqrt_bwd_f16, SqrtKernelOp,
        sqrtg(x),
        recipg(y + y))

UNARY_OP(float, sqrt_fwd_f32, sqrt_bwd_f32, SqrtKernelOp,
        sqrtg(x),
        recipg(y + y))

UNARY_OP(double, sqrt_fwd_f64, sqrt_bwd_f64, SqrtKernelOp,
        sqrtg(x),
        recipg(y + y))
        