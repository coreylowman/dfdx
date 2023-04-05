use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};
use num_traits::Float;

impl<F: Float> BinaryDerivative<F> for super::BinaryAddKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x + y
    }
    #[inline(always)]
    fn dfdx(&self, _: &F, _: &F) -> F {
        F::one()
    }
    #[inline(always)]
    fn dfdy(&self, _: &F, _: &F) -> F {
        F::one()
    }
}

impl<F: Float> UnaryDerivative<F> for super::ScalarAddKernelOp<F> {
    const DF_USES_FX: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x + self.scalar
    }
    #[inline(always)]
    fn df(&self, _: &F) -> F {
        F::one()
    }
}
