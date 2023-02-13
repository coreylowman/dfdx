use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SinKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.sin()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        x.cos()
    }
}
