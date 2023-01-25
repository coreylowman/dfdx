#include "unary_op_macros.cuh"

struct NansToKernelOp {
    float x;
};

UNARY_OP(nans_to_forward, nans_to_backward, NansToKernelOp,
        isnan(x) ? op.x : x,
        isnan(x) ? 0.0 : 1.0)
