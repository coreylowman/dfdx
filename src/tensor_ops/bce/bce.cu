#include "binary_op_macros.cuh"

struct BCEKernelOp {};

template<typename T>
__device__ T op_f(T logit, T prob) {
    T zero = 0.0;
    T one = 1.0;
    return maxg(logit, zero) - logit * prob + logg(one + expg(-absg(logit)));
}

template<typename T>
__device__ T op_dfdx(T logit, T prob) {
    T one = 1.0;
    return one - prob - one / (one + expg(logit));
}

template<typename T>
__device__ T op_dfdy(T logit, T prob) {
    return -logit;
}

BINARY_OP(__half, bce_fwd_f16, bce_bwd_lhs_f16, bce_bwd_rhs_f16, BCEKernelOp,
    __float2half(op_f(__half2float(x), __half2float(y))),
    op_dfdx(x, y),
    op_dfdy(x, y)
)

BINARY_OP(float, bce_fwd_f32, bce_bwd_lhs_f32, bce_bwd_rhs_f32, BCEKernelOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)

BINARY_OP(double, bce_fwd_f64, bce_bwd_lhs_f64, bce_bwd_rhs_f64, BCEKernelOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)
