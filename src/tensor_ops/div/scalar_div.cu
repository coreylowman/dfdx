#include "unary_op_macros.cuh"

template<typename F>
struct ScalarDivKernelOp {
    F scalar;
};

UNARY_OP(float, scalar_div_forward_f32, scalar_div_backward_f32, ScalarDivKernelOp<float>,
    x / op.scalar,
    1.0 / op.scalar);

UNARY_OP(double, scalar_div_forward_f64, scalar_div_backward_f64, ScalarDivKernelOp<double>,
    x / op.scalar,
    1.0 / op.scalar);
