#include "unary_op_macros.cuh"

struct ExpKernelOp {};

UNARY_OP(float, exp_fwd_f32, exp_bwd_f32, ExpKernelOp,
        expf(x),
        expf(x))

UNARY_OP(double, exp_fwd_f64, exp_bwd_f64, ExpKernelOp,
        exp(x),
        exp(x))
        