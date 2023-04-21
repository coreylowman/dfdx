use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::NegateKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = true;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.neg()
    }
    #[inline(always)]
    fn df(&self, _: &F) -> F {
        F::one().neg()
    }
    #[inline(always)]
    fn const_df(&self) -> F {
        F::one().neg()
    }
}
