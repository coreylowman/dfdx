#include "unary_op_macros.cuh"

struct ReLUKernelOp {};

UNARY_OP(relu_forward, relu_backward, ReLUKernelOp,
        fmaxf(x, 0.0),
        x > 0.0 ? 1.0 : 0.0)
