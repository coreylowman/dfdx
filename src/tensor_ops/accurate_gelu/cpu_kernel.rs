use crate::tensor_ops::cpu_kernels::UnaryDerivative;
#[cfg(feature = "f16")]
use half::f16;
use libm::{erf, erff};
use num_traits::{Float, FloatConst};

trait Erf {
    fn erf(self) -> Self;
}

#[cfg(feature = "f16")]
impl Erf for crate::dtypes::AMP<f16> {
    fn erf(self) -> Self {
        crate::dtypes::AMP(f16::from_f32(erff(self.0.to_f32())))
    }
}

#[cfg(feature = "f16")]
impl Erf for f16 {
    fn erf(self) -> Self {
        f16::from_f32(erff(self.to_f32()))
    }
}

impl Erf for f64 {
    fn erf(self) -> Self {
        erf(self)
    }
}

impl Erf for f32 {
    fn erf(self) -> Self {
        erff(self)
    }
}

impl<F: Float + FloatConst + Erf> UnaryDerivative<F> for super::AccurateGeLUKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        let alpha = F::FRAC_1_SQRT_2();
        F::from(0.5).unwrap() * x * (F::one() + (x * alpha).erf())
    }

    #[inline(always)]
    fn df(&self, &x: &F) -> F {
        let half = F::from(0.5).unwrap();
        let alpha = F::FRAC_1_SQRT_2();
        let x_sq = x * x;
        let normal_dist = F::FRAC_2_SQRT_PI() * (F::from(0.5).unwrap() * x_sq.neg()).exp();

        let left = half * x;
        let right = F::one() + (alpha * x).erf();

        let left_derivative = half * right;

        let right_derivative = left * normal_dist;

        left_derivative + right_derivative
    }
}
