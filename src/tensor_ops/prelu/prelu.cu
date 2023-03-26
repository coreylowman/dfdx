#include "binary_op_macros.cuh"
#include "unary_op_macros.cuh"

struct PReLUOp {};

template<typename F>
struct LeakyReLUOp {
    F slope;
};

UNARY_OP(float, lrelu_fwd_f32, lrelu_bwd_f32, LeakyReLUOp<float>,
    max(x, 0.0) + min(x,0.0)*op.slope,
    x >= 0 ? 1.0 : op.slope);

UNARY_OP(double, lrelu_fwd_f64, lrelu_bwd_f64, LeakyReLUOp<double>,
    max(x, 0.0) + min(x,0.0)*op.slope,
    x >= 0 ? 1.0 : op.slope);

BINARY_OP(float, prelu_fwd_f32, prelu_bwd_lhs_f32, prelu_bwd_rhs_f32, PReLUOp,
    max(x, 0.0) + min(x,0.0)*y,
    x >= 0 ? 1.0 : y,
    x >= 0 ? 0.0 : x);

BINARY_OP(double, prelu_fwd_f64, prelu_bwd_lhs_f64, prelu_bwd_rhs_f64, PReLUOp,
    max(x, 0.0) + min(x,0.0)*y,
    x >= 0 ? 1.0 : y,
    x >= 0 ? 0.0 : x);