#include "unary_op_macros.cuh"

struct NegateKernelOp {};

UNARY_OP(negate_forward, negate_backward, NegateKernelOp,
        -x,
        -1.0)
