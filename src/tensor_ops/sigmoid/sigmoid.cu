#include "unary_op_macros.cuh"
#define SIGMOID(X) (1.0 / (1.0 + expf(-X))) 

struct SigmoidKernelOp {};

UNARY_OP(sigmoid_forward, sigmoid_backward, SigmoidKernelOp,
        SIGMOID(x),
        SIGMOID(x) * (1.0 - SIGMOID(x)))
