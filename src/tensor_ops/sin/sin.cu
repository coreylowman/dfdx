#include "unary_op_macros.cuh"

struct SinKernelOp {};

UNARY_OP(__half, sin_fwd_f16, sin_bwd_f16, SinKernelOp,
        sing(x),
        cosg(x))

UNARY_OP(float, sin_fwd_f32, sin_bwd_f32, SinKernelOp,
        sing(x),
        cosg(x))

UNARY_OP(double, sin_fwd_f64, sin_bwd_f64, SinKernelOp,
        sing(x),
        cosg(x))
        