#include "unary_op_macros.cuh"
#include "cuda_utils.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

struct GeLUKernelOp {};

template<typename T>
__device__ T gelu_forward(T x) {
    constexpr T fastCoeff = 0.044715;
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + fastCoeff * x_cube;
    return 0.5 * x * (1.0 + tanhg(M_2_SQRTPI * M_SQRT1_2 * alpha));
}

template<typename T>
__device__ T gelu_backward(T x) {
    constexpr T kBeta = M_2_SQRTPI * M_SQRT2 * 0.5;                       
    constexpr T fastCoeff = 0.044715;
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T inner = kBeta * (x + fastCoeff * x_cube);
    T tanh_inner = tanhg(inner);

    T left = 0.5 * x;
    T right = 1.0 + tanh_inner;
    
    T left_derivative = 0.5 * right;

    T tanh_derivative = 1.0 - tanh_inner * tanh_inner;
    T inner_derivative = kBeta * (1.0 + 3.0 * fastCoeff * x_sq);
    T right_derivative = left * tanh_derivative * inner_derivative;
    return left_derivative + right_derivative;
}

UNARY_OP(float, gelu_forward_f32, gelu_backward_f32, GeLUKernelOp,
    gelu_forward(x),
    gelu_backward(x)
)

UNARY_OP(double, gelu_forward_f64, gelu_backward_f64, GeLUKernelOp,
    gelu_forward(x),
    gelu_backward(x)
)
