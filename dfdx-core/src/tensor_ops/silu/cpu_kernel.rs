use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SiLUKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;

    // x / (1 + e^-x)
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        *x / (F::one() + x.neg().exp())
    }

    // (1 + e^-x + x * e^-x) / (1 + e^-x)^2
    // alternative: (e^x (1 + e^x + x)) / (1 + e^x)^2
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        let exp_nx = x.neg().exp();
        (F::one() + exp_nx + *x * exp_nx) / (F::one() + exp_nx).powi(2)
    }
}
