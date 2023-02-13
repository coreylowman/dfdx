use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::LnKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.ln()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        x.recip()
    }
}
