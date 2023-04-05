use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SigmoidKernelOp {
    const DF_USES_FX: bool = true;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        F::one() / (F::one() + x.neg().exp())
    }
    #[inline(always)]
    fn df(&self, &fx: &F) -> F {
        fx * (F::one() - fx)
    }
}
