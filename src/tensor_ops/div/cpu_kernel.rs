use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::ScalarDivKernelOp<F> {
    fn f(&self, &x: &F) -> F {
        x / self.scalar
    }
    fn df(&self, _: &F) -> F {
        F::one() / self.scalar
    }
}

impl<F: Float> BinaryDerivative<F> for super::BinaryDivKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x / y
    }
    #[inline(always)]
    fn dfdx(&self, _: &F, &y: &F) -> F {
        F::one() / y
    }
    #[inline(always)]
    fn dfdy(&self, &x: &F, y: &F) -> F {
        -x / y.powi(2)
    }
}
