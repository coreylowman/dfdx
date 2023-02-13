#include "unary_op_macros.cuh"

template<typename T>
struct ScalarDivKernelOp {
    T scalar;
};

UNARY_OP(float, sdiv_fwd_f32, sdiv_bwd_f32, ScalarDivKernelOp<float>,
    x / op.scalar,
    1.0 / op.scalar);

UNARY_OP(double, sdiv_fwd_f64, sdiv_bwd_f64, ScalarDivKernelOp<double>,
    x / op.scalar,
    1.0 / op.scalar);
