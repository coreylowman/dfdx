#include "unary_op_macros.cuh"

template<typename F>
struct PowFKernelOp {
    F rhs;
};

UNARY_OP(float, pow_forward_f32, pow_backward_f32, PowFKernelOp<float>,
        powf(x, op.rhs),
        op.rhs * powf(x, op.rhs - 1.0))
