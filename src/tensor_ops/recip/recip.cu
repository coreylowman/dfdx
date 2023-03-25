#include "unary_op_macros.cuh"

struct RecipKernelOp {};

UNARY_OP(
    float, recip_fwd_f32, recip_bwd_f32, RecipKernelOp,
    1 / x,
    -1 / (x * x)
)

UNARY_OP(
    double, recip_fwd_f64, recip_bwd_f64, RecipKernelOp,
    1 / x,
    -1 / (x * x)
)
