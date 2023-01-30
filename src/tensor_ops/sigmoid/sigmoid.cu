#include "unary_op_macros.cuh"
#define SIGMOID_f32(X) (1.0 / (1.0 + expf(-X))) 

struct SigmoidKernelOp {};

UNARY_OP(float, sigmoid_forward_f32, sigmoid_backward_f32, SigmoidKernelOp,
        SIGMOID_f32(x),
        SIGMOID_f32(x) * (1.0 - SIGMOID_f32(x)))
