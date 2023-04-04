use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::ReLUKernelOp {
    const DF_USES_FX: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.max(F::zero())
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if x > &F::zero() {
            F::one()
        } else {
            F::zero()
        }
    }
}
