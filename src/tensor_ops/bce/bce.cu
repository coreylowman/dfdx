#include "binary_op_macros.cuh"

struct BCEKernelOp {};

LONG_BINARY_OP(bce_forward, bce_backward, BCEKernelOp,
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
