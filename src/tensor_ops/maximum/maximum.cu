#include "binary_op_macros.cuh"

struct MaximumKernalOp {};

LONG_BINARY_OP(maximum_forward, maximum_backward, MaximumKernalOp,
    {
        fx = fmaxf(x, y);
    },
    {
        if (x > y) {
            dfdx = 1.0;
            dfdy = 0.0;
        } else if (x < y) {
            dfdx = 0.0;
            dfdy = 1.0;
        } else {
            dfdx = 0.5;
            dfdy = 0.5;
        }
    }
)
