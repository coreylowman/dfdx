use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use no_std_compat::f32::consts::PI;

impl UnaryDerivative<f32> for super::GeLUKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        let alpha = x + 0.044715 * x.powf(3.0);
        0.5 * (*x) * (1.0 + f32::tanh((2.0f32 / PI).sqrt() * alpha))
    }

    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        let sqrt2 = 2.0f32.sqrt();
        let sqrt_2_pi = 2.0 * (1.0 / PI).sqrt();
        let beta = sqrt2 * sqrt_2_pi * 0.5;
        let kappa = 0.044715;
        let x_sq = x * x;
        let x_cube = x_sq * x;
        let inner = beta * (x + kappa * x_cube);
        let tanh_inner = f32::tanh(inner);

        let left = 0.5 * x;
        let right = 1.0 + tanh_inner;

        let left_derivative = 0.5 * right;

        let tanh_derivative = 1.0 - tanh_inner * tanh_inner;
        let inner_derivative = beta * (1.0 + 3.0 * kappa * x_sq);
        let right_derivative = left * tanh_derivative * inner_derivative;

        left_derivative + right_derivative
    }
}
