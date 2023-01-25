#include "unary_op_macros.cuh"

struct ScalarDivKernelOp {
    float scalar;
};

UNARY_OP(scalar_div_forward, scalar_div_backward, ScalarDivKernelOp,
        x / op.scalar,
        1.0 / op.scalar);
