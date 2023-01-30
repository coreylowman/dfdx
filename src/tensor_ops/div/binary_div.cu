#include "binary_op_macros.cuh"

struct BinaryDivOp {};

BINARY_OP(float, binary_div_forward_f32, binary_div_backward_f32, BinaryDivOp,
         x / y,
         1.0 / y,
         -x / (y * y))
