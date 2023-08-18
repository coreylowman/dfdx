#include "unary_op_macros.cuh"

template<typename F>
struct ScalarSubKernelOp {
    F scalar;
};

UNARY_OP(__half, ssub_fwd_f16, ssub_bwd_f16, ScalarSubKernelOp<__half>,
    x - op.scalar,
    1.0);

UNARY_OP(float, ssub_fwd_f32, ssub_bwd_f32, ScalarSubKernelOp<float>,
        x - op.scalar,
        1.0);

UNARY_OP(double, ssub_fwd_f64, ssub_bwd_f64, ScalarSubKernelOp<double>,
    x - op.scalar,
    1.0);
    