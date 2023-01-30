use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::NansToKernelOp<F> {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        if x.is_nan() {
            self.0
        } else {
            *x
        }
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if x.is_nan() {
            F::zero()
        } else {
            F::one()
        }
    }
}
