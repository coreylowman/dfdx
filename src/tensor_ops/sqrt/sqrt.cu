#include "unary_op_macros.cuh"

struct SqrtKernelOp {};

UNARY_OP(sqrt_forward, sqrt_backward, SqrtKernelOp,
        sqrtf(x),
        0.5 / sqrtf(x))
