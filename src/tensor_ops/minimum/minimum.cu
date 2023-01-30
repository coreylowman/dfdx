#include "binary_op_macros.cuh"

struct MinimumKernelOp {};

LONG_BINARY_OP(float, minimum_forward_f32, minimum_backward_f32, MinimumKernelOp,
    {
        fx = fminf(x, y);
    },
    {
        if (x < y) {
            dfdx = 1.0;
            dfdy = 0.0;
        } else if (x > y) {
            dfdx = 0.0;
            dfdy = 1.0;
        } else {
            dfdx = 0.5;
            dfdy = 0.5;
        }
    }
)
