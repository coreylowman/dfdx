#include "unary_op_macros.cuh"

struct SinKernelOp {};

UNARY_OP(float, sin_forward_f32, sin_backward_f32, SinKernelOp,
        sinf(x),
        cosf(x))
