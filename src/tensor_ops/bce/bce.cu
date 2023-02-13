#include "binary_op_macros.cuh"

struct BCEKernelOp {};

template<typename T>
__device__ T op_f(T logit, T prob) {
    return maxg(logit, 0.0) - logit * prob + logg(1.0 + expg(-absg(logit)));
}

template<typename T>
__device__ T op_dfdx(T logit, T prob) {
    return 1.0 - prob - 1 / (1.0 + expg(logit));
}

template<typename T>
__device__ T op_dfdy(T logit, T prob) {
    return -logit;
}

BINARY_OP(float, bce_fwd_f32, bce_bwd_f32, BCEKernelOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)

BINARY_OP(double, bce_fwd_f64, bce_bwd_f64, BCEKernelOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)
