#include "unary_op_macros.cuh"

template<typename F>
struct ScalarAddKernelOp {
    F scalar;
};

UNARY_OP(float, scalar_add_forward_f32, scalar_add_backward_f32, ScalarAddKernelOp<float>,
    x + op.scalar,
    1.0);

UNARY_OP(double, scalar_add_forward_f64, scalar_add_backward_f64, ScalarAddKernelOp<double>,
    x + op.scalar,
    1.0);
    