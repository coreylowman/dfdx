#include "unary_op_macros.cuh"

struct CosKernelOp {};

UNARY_OP(cos_forward, cos_backward, CosKernelOp,
        cosf(x),
        -sinf(x))
