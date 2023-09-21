#include "unary_op_macros.cuh"

struct RecipKernelOp {};

UNARY_OP(
    __half, recip_fwd_f16, recip_bwd_f16, RecipKernelOp,
    recipg(x),
    -y * y
)

UNARY_OP(
    float, recip_fwd_f32, recip_bwd_f32, RecipKernelOp,
    recipg(x),
    -y * y
)

UNARY_OP(
    double, recip_fwd_f64, recip_bwd_f64, RecipKernelOp,
    recipg(x),
    -y * y
)
