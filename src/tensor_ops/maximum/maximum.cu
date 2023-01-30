#include "binary_op_macros.cuh"

struct MaximumKernalOp {};

LONG_BINARY_OP(float, maximum_forward_f32, maximum_backward_f32, MaximumKernalOp,
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
