#include "unary_op_macros.cuh"

struct SquareKernelOp {};

UNARY_OP(float, square_forward_f32, square_backward_f32, SquareKernelOp,
        x * x,
        2.0 * x)

UNARY_OP(double, square_forward_f64, square_backward_f64, SquareKernelOp,
        x * x,
        2.0 * x)
        