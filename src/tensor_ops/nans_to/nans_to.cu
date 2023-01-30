#include "unary_op_macros.cuh"

template<typename F>
struct NansToKernelOp {
    F x;
};

UNARY_OP(float, nans_to_forward_f32, nans_to_backward_f32, NansToKernelOp<float>,
        isnan(x) ? op.x : x,
        isnan(x) ? 0.0 : 1.0)
