#include "binary_op_macros.cuh"

struct BinarySubKernelOp {};

BINARY_OP(float, binary_sub_forward_f32, binary_sub_backward_f32, BinarySubKernelOp,
         x - y,
         1.0,
         -1.0)
