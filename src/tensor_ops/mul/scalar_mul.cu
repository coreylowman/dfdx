#include "unary_op_macros.cuh"

template<typename F>
struct ScalarMulKernelOp {
    F scalar;
};

UNARY_OP(float, scalar_mul_forward_f32, scalar_mul_backward_f32, ScalarMulKernelOp<float>,
        x * op.scalar,
        op.scalar);
