#include "unary_op_macros.cuh"

struct ScalarSubKernelOp {
    float scalar;
};

UNARY_OP(scalar_sub_forward, scalar_sub_backward, ScalarSubKernelOp,
        x - op.scalar,
        1.0);
