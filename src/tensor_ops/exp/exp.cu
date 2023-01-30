#include "unary_op_macros.cuh"

struct ExpKernelOp {};

UNARY_OP(float, exp_forward_f32, exp_backward_f32, ExpKernelOp,
        expf(x),
        expf(x))

UNARY_OP(double, exp_forward_f64, exp_backward_f64, ExpKernelOp,
        exp(x),
        exp(x))
        