#include "binary_op_macros.cuh"

struct BinaryMulKernalOp {};

BINARY_OP(binary_mul_forward, binary_mul_backward, BinaryMulKernalOp,
         x * y,
         y,
         x)
