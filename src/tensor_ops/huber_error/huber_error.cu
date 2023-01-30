#include "binary_op_macros.cuh"

struct HuberErrorOp {
    float delta;
};

LONG_BINARY_OP(float, huber_error_forward_f32, huber_error_backward_f32, HuberErrorOp,
    {
        float a = x - y;

        if (fabsf(a) < op.delta) {
            fx = a * a * 0.5;
        } else {
            fx = op.delta * (fabsf(a) - 0.5 * op.delta);
        }
    },
    {
        auto a = x - y;

        if (a == 0.0) {
            dfdx = 0.0;
        } else if (fabsf(a) < op.delta) {
            dfdx = a;
        } else {
            dfdx = copysignf(op.delta, a);
        }

        dfdy = -dfdx;
    }
)
