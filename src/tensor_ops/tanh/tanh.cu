#include "unary_op_macros.cuh"

struct TanhKernelOp {};

UNARY_OP(float, tanh_fwd_f32, tanh_bwd_f32, TanhKernelOp,
        tanhf(x),
        1 - y * y)

UNARY_OP(double, tanh_fwd_f64, tanh_bwd_f64, TanhKernelOp,
        tanh(x),
        1 - y * y)
        