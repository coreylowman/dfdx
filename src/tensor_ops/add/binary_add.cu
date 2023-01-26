#include "binary_op_macros.cuh"

struct BinaryAddOp {};

BINARY_OP(binary_add_forward, binary_add_backward, BinaryAddOp,
        x + y,
        1.0,
        1.0)
