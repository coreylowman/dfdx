#include "unary_op_macros.cuh"

template<typename F>
struct NansToKernelOp {
    F x;
};

UNARY_OP(__half, nans_to_fwd_f16, nans_to_bwd_f16, NansToKernelOp<__half>,
    isnang(x) ? op.x : x,
    isnang(x) ? 0.0 : 1.0)

UNARY_OP(float, nans_to_fwd_f32, nans_to_bwd_f32, NansToKernelOp<float>,
    isnang(x) ? op.x : x,
    isnang(x) ? 0.0 : 1.0)

UNARY_OP(double, nans_to_fwd_f64, nans_to_bwd_f64, NansToKernelOp<double>,
    isnang(x) ? op.x : x,
    isnang(x) ? 0.0 : 1.0)
    