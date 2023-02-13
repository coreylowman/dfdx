#include "binary_op_macros.cuh"

struct BinarySubKernelOp {};

BINARY_OP(float, bsub_fwd_f32, bsub_bwd_f32, BinarySubKernelOp,
    x - y,
    1.0,
    -1.0)

BINARY_OP(double, bsub_fwd_f64, bsub_bwd_f64, BinarySubKernelOp,
    x - y,
    1.0,
    -1.0)
   