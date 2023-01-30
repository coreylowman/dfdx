#include "binary_op_macros.cuh"

struct BinaryMulKernalOp {};

BINARY_OP(float, binary_mul_forward_f32, binary_mul_backward_f32, BinaryMulKernalOp,
         x * y,
         y,
         x)
