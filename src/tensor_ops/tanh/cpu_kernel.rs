use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::TanhKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.tanh()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        F::one() - x.tanh().powi(2)
    }
}
