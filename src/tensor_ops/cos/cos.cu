#include "unary_op_macros.cuh"

struct CosKernelOp {};

UNARY_OP(float, cos_forward_f32, cos_backward_f32, CosKernelOp,
        cosf(x),
        -sinf(x))
