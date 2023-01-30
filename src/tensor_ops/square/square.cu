#include "unary_op_macros.cuh"

struct SquareKernelOp {};

UNARY_OP(float, square_forward_f32, square_backward_f32, SquareKernelOp,
        x * x,
        2.0 * x)
