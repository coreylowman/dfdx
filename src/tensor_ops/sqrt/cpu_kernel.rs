use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SqrtKernelOp {
    const DF_USES_FX: bool = true;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.sqrt()
    }
    #[inline(always)]
    fn df(&self, &fx: &F) -> F {
        (fx + fx).recip()
    }
}
