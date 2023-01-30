#include "unary_op_macros.cuh"

struct NegateKernelOp {};

UNARY_OP(float, negate_forward_f32, negate_backward_f32, NegateKernelOp,
        -x,
        -1.0)
