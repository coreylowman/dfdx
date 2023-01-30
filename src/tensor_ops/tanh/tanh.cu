#include "unary_op_macros.cuh"

struct TanhKernelOp {};

UNARY_OP(float, tanh_forward_f32, tanh_backward_f32, TanhKernelOp,
        tanhf(x),
        1 - tanhf(x) * tanhf(x))

UNARY_OP(double, tanh_forward_f64, tanh_backward_f64, TanhKernelOp,
        tanh(x),
        1 - tanh(x) * tanh(x))
        