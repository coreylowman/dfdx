#include "unary_op_macros.cuh"

struct SinKernelOp {};

UNARY_OP(float, sin_fwd_f32, sin_bwd_f32, SinKernelOp,
        sinf(x),
        cosf(x))

UNARY_OP(double, sin_fwd_f64, sin_bwd_f64, SinKernelOp,
        sin(x),
        cos(x))
        