#include "binary_op_macros.cuh"

struct BinaryAddOp {};

BINARY_OP(float, binary_add_forward_f32, binary_add_backward_f32, BinaryAddOp,
        x + y,
        1.0,
        1.0)
