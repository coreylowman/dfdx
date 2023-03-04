#include "binary_op_macros.cuh"

struct BinaryDivOp {};

BINARY_OP(float, bdiv_fwd_f32, bdiv_bwd_lhs_f32, bdiv_bwd_rhs_f32, BinaryDivOp,
    x / y,
    1.0 / y,
    -x / (y * y))

BINARY_OP(double, bdiv_fwd_f64, bdiv_bwd_lhs_f64, bdiv_bwd_rhs_f64, BinaryDivOp,
    x / y,
    1.0 / y,
    -x / (y * y))
   
