#include "binary_op_macros.cuh"

template<typename T>
struct HuberErrorOp {
    T delta;
};

template<typename T>
__device__ T op_f(HuberErrorOp<T> op, T x, T y) {
    T a = x - y;
    T half = 0.5;
    if (absg(a) < op.delta) {
        return a * a * half;
    } else {
        return op.delta * (absg(a) - half * op.delta);
    }
}

template<typename T>
__device__ T op_dfdx(HuberErrorOp<T> op, T x, T y) {
    T a = x - y;
    T zero = 0.0;
    if (a == zero) {
        return zero;
    } else if (absg(a) < op.delta) {
        return a;
    } else {
        return copysigng(op.delta, a);
    }
}

template<typename T>
__device__ T op_dfdy(HuberErrorOp<T> op, T x, T y) {
    return -op_dfdx(op, x, y);
}

BINARY_OP(__half, huber_fwd_f16, huber_bwd_lhs_f16, huber_bwd_rhs_f16, HuberErrorOp<__half>,
    op_f(op, x, y),
    op_dfdx(op, x, y),
    op_dfdy(op, x, y)
)

BINARY_OP(float, huber_fwd_f32, huber_bwd_lhs_f32, huber_bwd_rhs_f32, HuberErrorOp<float>,
    op_f(op, x, y),
    op_dfdx(op, x, y),
    op_dfdy(op, x, y)
)

BINARY_OP(double, huber_fwd_f64, huber_bwd_lhs_f64, huber_bwd_rhs_f64, HuberErrorOp<double>,
    op_f(op, x, y),
    op_dfdx(op, x, y),
    op_dfdy(op, x, y)
)
