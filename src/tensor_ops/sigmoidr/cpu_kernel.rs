use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SigmoidrKernelOp {
    const DF_USES_FX: bool = true;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        F::one() / (F::one() + x.neg().exp())
    }
    #[inline(always)]
    fn df(&self, &fx: &F) -> F {
        let d = fx * (F::one() - fx);
        F::max(d, F::from(0.0000001).unwrap())
    }
}
