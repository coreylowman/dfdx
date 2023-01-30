use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::AbsKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.abs()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if x == &F::zero() {
            F::zero()
        } else {
            x.signum()
        }
    }
}
