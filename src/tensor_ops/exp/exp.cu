#include "unary_op_macros.cuh"

struct ExpKernelOp {};

UNARY_OP(exp_forward, exp_backward, ExpKernelOp,
        expf(x),
        expf(x))
