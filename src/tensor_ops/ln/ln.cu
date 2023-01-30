#include "unary_op_macros.cuh"

struct LnKernelOp {};

UNARY_OP(float, ln_forward_f32, ln_backward_f32, LnKernelOp,
        logf(x),
        1.0 / x)

UNARY_OP(double, ln_forward_f64, ln_backward_f64, LnKernelOp,
        log(x),
        1.0 / x)
        