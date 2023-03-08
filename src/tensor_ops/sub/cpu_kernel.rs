use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};

impl<F: num_traits::Float> UnaryDerivative<F> for super::ScalarSubKernelOp<F> {
    fn f(&self, &x: &F) -> F {
        x - self.scalar
    }
    fn df(&self, _: &F) -> F {
        F::one()
    }
}

impl<F: num_traits::Float> BinaryDerivative<F> for super::BinarySubKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x - y
    }
    #[inline(always)]
    fn dfdx(&self, _: &F, _: &F) -> F {
        F::one()
    }
    #[inline(always)]
    fn dfdy(&self, _: &F, _: &F) -> F {
        -F::one()
    }
}
