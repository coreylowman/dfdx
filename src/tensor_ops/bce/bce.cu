#include "binary_op_macros.cuh"

struct BCEKernelOp {};

LONG_BINARY_OP(float, bce_forward_f32, bce_backward_f32, BCEKernelOp,
    {
        float logit = lhs[lhs_i];
        float prob = rhs[rhs_i];

        fx = fmaxf(logit, 0.0) - logit * prob + logf(1.0 + expf(-fabsf(logit)));
    },
    {
        auto logit = lhs[lhs_i];
        auto prob = rhs[rhs_i];
        auto go = grad_out[out_i];

        dfdx = 1.0 - prob - 1 / (1.0 + expf(logit));
        dfdy = -logit;
    }
)

LONG_BINARY_OP(double, bce_forward_f64, bce_backward_f64, BCEKernelOp,
    {
        auto logit = lhs[lhs_i];
        auto prob = rhs[rhs_i];

        fx = fmax(logit, 0.0) - logit * prob + log(1.0 + exp(-fabs(logit)));
    },
    {
        auto logit = lhs[lhs_i];
        auto prob = rhs[rhs_i];
        auto go = grad_out[out_i];

        dfdx = 1.0 - prob - 1 / (1.0 + exp(logit));
        dfdy = -logit;
    }
)
