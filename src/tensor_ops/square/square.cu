#include "unary_op_macros.cuh"

struct SquareKernelOp {};

UNARY_OP(float, square_fwd_f32, square_bwd_f32, SquareKernelOp,
        x * x,
        2.0 * x)

UNARY_OP(double, square_fwd_f64, square_bwd_f64, SquareKernelOp,
        x * x,
        2.0 * x)
        