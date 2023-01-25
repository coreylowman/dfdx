#include "unary_op_macros.cuh"

struct TanhKernelOp {};

UNARY_OP(tanh_forward, tanh_backward, TanhKernelOp,
        tanhf(x),
        1 - tanhf(x) * tanhf(x))
