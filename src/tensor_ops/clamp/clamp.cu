#include "unary_op_macros.cuh"

struct ClampKernelOp {
    float min;
    float max;
};

UNARY_OP(clamp_forward, clamp_backward, ClampKernelOp,
        fmaxf(fminf(x, op.max), op.min),
        x <= op.max && x >= op.min ? 1.0 : 0.0)
