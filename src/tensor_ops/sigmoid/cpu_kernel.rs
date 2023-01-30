use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::SigmoidKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        F::one() / (F::one() + x.neg().exp())
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        let fx = self.f(x);
        fx * (F::one() - fx)
    }
}
