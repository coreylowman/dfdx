use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::{Float, FloatConst};

impl<F: Float + FloatConst> UnaryDerivative<F> for super::GeLUKernelOp {
    const DF_USES_FX: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        let alpha = x + F::from(0.044715).unwrap() * x.powi(3);
        F::from(0.5).unwrap() * x * (F::one() + (F::FRAC_2_PI().sqrt() * alpha).tanh())
    }

    #[inline(always)]
    fn df(&self, &x: &F) -> F {
        let half = F::from(0.5).unwrap();
        let three = F::from(3.0).unwrap();
        let beta = F::SQRT_2() * F::FRAC_2_SQRT_PI() * half;
        let kappa = F::from(0.044715).unwrap();
        let x_sq = x * x;
        let x_cube = x_sq * x;
        let tanh_inner = (beta * (x + kappa * x_cube)).tanh();

        let left = half * x;
        let right = F::one() + tanh_inner;

        let left_derivative = half * right;

        let tanh_derivative = F::one() - tanh_inner * tanh_inner;
        let inner_derivative = beta * (F::one() + three * kappa * x_sq);
        let right_derivative = left * tanh_derivative * inner_derivative;

        left_derivative + right_derivative
    }
}
