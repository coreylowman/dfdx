#include "unary_op_macros.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

struct GeLUKernelOp {};

template <typename T> __device__ T gelu_fwd(T x) {
  T fastCoeff = 0.044715;
  T one = 1.0;
  T half = 0.5;
  T beta = M_2_SQRTPI * M_SQRT1_2;
  T x_sq = x * x;
  T x_cube = x_sq * x;
  T alpha = x + fastCoeff * x_cube;
  return half * x * (one + tanhg(beta * alpha));
}

template <typename T> __device__ T gelu_bwd(T x) {
  T one = 1.0;
  T three = 3.0;
  T half = 0.5;
  T fastCoeff = 0.044715;
  T kBeta = M_2_SQRTPI * M_SQRT2 * 0.5;
  T x_sq = x * x;
  T x_cube = x_sq * x;
  T inner = kBeta * (x + fastCoeff * x_cube);
  T tanh_inner = tanhg(inner);

  T left = half * x;
  T right = one + tanh_inner;

  T left_derivative = half * right;

  T tanh_derivative = one - tanh_inner * tanh_inner;
  T inner_derivative = kBeta * (one + three * fastCoeff * x_sq);
  T right_derivative = left * tanh_derivative * inner_derivative;
  return left_derivative + right_derivative;
}

UNARY_OP(__half, gelu_fwd_f16, gelu_bwd_f16, GeLUKernelOp, gelu_fwd(x),
         gelu_bwd(x))

UNARY_OP(float, gelu_fwd_f32, gelu_bwd_f32, GeLUKernelOp, gelu_fwd(x),
         gelu_bwd(x))

UNARY_OP(double, gelu_fwd_f64, gelu_bwd_f64, GeLUKernelOp, gelu_fwd(x),
         gelu_bwd(x))
