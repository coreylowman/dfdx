#include "unary_op_macros.cuh"

struct PowFKernelOp {
    float rhs;
};

UNARY_OP(pow_forward, pow_backward, PowFKernelOp,
        powf(x, op.rhs),
        op.rhs * powf(x, op.rhs - 1.0))
