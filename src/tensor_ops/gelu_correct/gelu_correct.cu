#include "unary_op_macros.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

struct GeLUKernelOp {};

template <typename T> __device__ T gelu_correct_fwd(T x) {
  T one = 1.0;
  T half = 0.5;
  T alpha = M_SQRT1_2;
  return half * x * (one + erfg(x * alpha))
}

template <typename T> __device__ T gelu_correct_bwd(T x) {
  T one = 1.0;
  T half = 0.5;
  T alpha = M_SQRT1_2;
  T x_sq = x * x;
  T norm = expg(M_2_SQRTPI * half * x_sq);

  T left = half * x;
  T right = one + erfg(alph * x);

  T left_derivative = half * right;

  T right_derivative = left * normal_dist;

  return left_derivative + right_derivative;
}

UNARY_OP(__half, gelu_correct_fwd_f16, gelu_correct_bwd_f16,
         GeLUCorrectKernelOp, gelu_correct_fwd(x), gelu_correct_bwd(x))

UNARY_OP(float, gelu_correct_fwd_f32, gelu_correct_bwd_f32, GeLUCorrectKernelOp,
         gelu_correct_fwd(x), gelu_correct_bwd(x))

UNARY_OP(double, gelu_correct_fwd_f64, gelu_correct_bwd_f64,
         GeLUCorrectKernelOp, gelu_correct_fwd(x), gelu_correct_bwd(x))
