#include "unary_op_macros.cuh"

struct LnKernelOp {};

UNARY_OP(ln_forward, ln_backward, LnKernelOp,
        logf(x),
        1.0 / x)
