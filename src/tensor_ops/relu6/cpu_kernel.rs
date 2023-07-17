use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::ReLU6KernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.max(F::zero()).min(F::from(6.0).unwrap())
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if x > &F::zero() && x < &F::from(6.0).unwrap() {
            F::one()
        } else {
            F::zero()
        }
    }
}
