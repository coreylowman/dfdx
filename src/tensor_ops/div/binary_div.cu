#include "binary_op_macros.cuh"

struct BinaryDivOp {};

BINARY_OP(binary_div_forward, binary_div_backward, BinaryDivOp,
         x / y,
         1.0 / y,
         -x / (y * y))
