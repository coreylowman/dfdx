#include "unary_op_macros.cuh"

struct ScalarAddKernelOp {
    float scalar;
};

UNARY_OP(scalar_add_forward, scalar_add_backward, ScalarAddKernelOp,
        x + op.scalar,
        1.0);
