#include "unary_op_macros.cuh"

struct TanhKernelOp {};

UNARY_OP(float, tanh_forward_f32, tanh_backward_f32, TanhKernelOp,
        tanhf(x),
        1 - tanhf(x) * tanhf(x))
