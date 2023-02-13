#include "binary_op_macros.cuh"

struct BinaryAddOp {};

BINARY_OP(float, badd_fwd_f32, badd_bwd_f32, BinaryAddOp,
    x + y,
    1.0,
    1.0)

BINARY_OP(double, badd_fwd_f64, badd_bwd_f64, BinaryAddOp,
    x + y,
    1.0,
    1.0)