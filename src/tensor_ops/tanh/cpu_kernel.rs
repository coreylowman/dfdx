use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::TanhKernelOp {
    const DF_USES_FX: bool = true;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.tanh()
    }
    #[inline(always)]
    fn df(&self, fx: &F) -> F {
        F::one() - fx.powi(2)
    }
}
