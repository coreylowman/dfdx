#include "unary_op_macros.cuh"

template<typename F>
struct ScalarMulKernelOp {
    F scalar;
};

UNARY_OP(__half, smul_fwd_f16, smul_bwd_f16, ScalarMulKernelOp<__half>,
    x * op.scalar,
    op.scalar);

UNARY_OP(float, smul_fwd_f32, smul_bwd_f32, ScalarMulKernelOp<float>,
    x * op.scalar,
    op.scalar);

UNARY_OP(double, smul_fwd_f64, smul_bwd_f64, ScalarMulKernelOp<double>,
    x * op.scalar,
    op.scalar);
    