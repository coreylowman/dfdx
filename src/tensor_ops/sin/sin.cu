#include "unary_op_macros.cuh"

struct SinKernelOp {};

UNARY_OP(float, sin_forward_f32, sin_backward_f32, SinKernelOp,
        sinf(x),
        cosf(x))

UNARY_OP(double, sin_forward_f64, sin_backward_f64, SinKernelOp,
        sin(x),
        cos(x))
        