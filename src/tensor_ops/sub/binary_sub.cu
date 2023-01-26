#include "binary_op_macros.cuh"

struct BinarySubKernelOp {};

BINARY_OP(binary_sub_forward, binary_sub_backward, BinarySubKernelOp,
         x - y,
         1.0,
         -1.0)
