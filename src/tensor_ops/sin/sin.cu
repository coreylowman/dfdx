#include "unary_op_macros.cuh"

struct SinKernelOp {};

UNARY_OP(sin_forward, sin_backward, SinKernelOp,
        sinf(x),
        cosf(x))
