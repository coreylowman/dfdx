#include "unary_op_macros.cuh"

struct ScalarMulKernelOp {
    float scalar;
};

UNARY_OP(scalar_mul_forward, scalar_mul_backward, ScalarMulKernelOp,
        x * op.scalar,
        op.scalar);
