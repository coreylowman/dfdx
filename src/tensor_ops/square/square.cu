#include "unary_op_macros.cuh"

struct SquareKernelOp {};

UNARY_OP(square_forward, square_backward, SquareKernelOp,
        x * x,
        2.0 * x)
