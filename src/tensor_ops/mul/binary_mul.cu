#include "binary_op_macros.cuh"

struct BinaryMulKernalOp {};

BINARY_OP(float, bmul_fwd_f32, bmul_bwd_lhs_f32, bmul_bwd_rhs_f32, BinaryMulKernalOp,
    x * y,
    y,
    x)

BINARY_OP(double, bmul_fwd_f64, bmul_bwd_lhs_f64, bmul_bwd_rhs_f64, BinaryMulKernalOp,
    x * y,
    y,
    x)
   
