#include "unary_op_macros.cuh"

#define SIGMOID_f32(X) (1.0 / (1.0 + expf(-X))) 
#define SIGMOID_f64(X) (1.0 / (1.0 + exp(-X))) 

struct SigmoidKernelOp {};

UNARY_OP(float, sigmoid_fwd_f32, sigmoid_bwd_f32, SigmoidKernelOp,
        SIGMOID_f32(x),
        SIGMOID_f32(x) * (1.0 - SIGMOID_f32(x)))

UNARY_OP(double, sigmoid_fwd_f64, sigmoid_bwd_f64, SigmoidKernelOp,
        SIGMOID_f64(x),
        SIGMOID_f64(x) * (1.0 - SIGMOID_f64(x)))
        