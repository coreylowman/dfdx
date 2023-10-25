#include "unary_op_macros.cuh"

template<typename T>
struct ScalarDivKernelOp {
    T scalar;
};

UNARY_OP(__half, sdiv_fwd_f16, sdiv_bwd_f16, ScalarDivKernelOp<__half>,
    x / op.scalar,
    recipg(op.scalar));

UNARY_OP(float, sdiv_fwd_f32, sdiv_bwd_f32, ScalarDivKernelOp<float>,
    x / op.scalar,
    recipg(op.scalar));

UNARY_OP(double, sdiv_fwd_f64, sdiv_bwd_f64, ScalarDivKernelOp<double>,
    x / op.scalar,
    recipg(op.scalar));
