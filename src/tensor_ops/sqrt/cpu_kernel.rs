use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SqrtKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.sqrt()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        F::from(0.5).unwrap() / x.sqrt()
    }
}
