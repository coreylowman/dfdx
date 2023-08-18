#include "unary_op_macros.cuh"

struct AbsKernelOp {};

UNARY_OP(__half, abs_fwd_f16, abs_bwd_f16, AbsKernelOp,
        absg(x),
        x == __float2half(0.0) ? __float2half(0.0) : copysigng(__float2half(1.0), x));

UNARY_OP(float, abs_fwd_f32, abs_bwd_f32, AbsKernelOp,
        absg(x),
        x == 0.0 ? 0.0f : copysigng(1.0f, x));

UNARY_OP(double, abs_fwd_f64, abs_bwd_f64, AbsKernelOp,
        absg(x),
        x == 0.0 ? 0.0 : copysigng(1.0, x));
