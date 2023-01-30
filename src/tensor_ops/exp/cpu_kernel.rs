use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::ExpKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.exp()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        x.exp()
    }
}
