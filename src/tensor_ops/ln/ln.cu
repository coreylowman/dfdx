#include "unary_op_macros.cuh"

struct LnKernelOp {};

UNARY_OP(float, ln_fwd_f32, ln_bwd_f32, LnKernelOp,
        logf(x),
        1.0 / x)

UNARY_OP(double, ln_fwd_f64, ln_bwd_f64, LnKernelOp,
        log(x),
        1.0 / x)
        