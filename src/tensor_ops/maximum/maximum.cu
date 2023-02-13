#include "binary_op_macros.cuh"

struct MaximumKernalOp {};

template<typename T>
__device__ T op_f(T x, T y) {
    return maxg(x, y);
}

template<typename T>
__device__ T op_dfdx(T x, T y) {
    return (x > y) ? 1.0 : ((x < y) ? 0.0 : 0.5);
}

template<typename T>
__device__ T op_dfdy(T x, T y) {
    return (x > y) ? 0.0 : ((x < y) ? 1.0 : 0.5);
}

BINARY_OP(float, maximum_fwd_f32, maximum_bwd_f32, MaximumKernalOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)

BINARY_OP(double, maximum_fwd_f64, maximum_bwd_f64, MaximumKernalOp,
    op_f(x, y),
    op_dfdx(x, y),
    op_dfdy(x, y)
)
