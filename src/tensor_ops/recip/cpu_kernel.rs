use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::RecipKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.recip()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        -x.powi(2).recip()
    }
}
