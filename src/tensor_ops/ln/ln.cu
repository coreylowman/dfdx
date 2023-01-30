#include "unary_op_macros.cuh"

struct LnKernelOp {};

UNARY_OP(float, ln_forward_f32, ln_backward_f32, LnKernelOp,
        logf(x),
        1.0 / x)
