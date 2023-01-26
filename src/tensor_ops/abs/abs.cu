#include "unary_op_macros.cuh"

struct AbsKernelOp {};

UNARY_OP(abs_forward, abs_backward, AbsKernelOp,
        fabsf(x),
        x == 0.0 ? 0.0 : copysignf(1.0, x));
