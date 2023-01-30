#include "unary_op_macros.cuh"

template<typename F>
struct ScalarSubKernelOp {
    F scalar;
};

UNARY_OP(float, scalar_sub_forward_f32, scalar_sub_backward_f32, ScalarSubKernelOp<float>,
        x - op.scalar,
        1.0);
