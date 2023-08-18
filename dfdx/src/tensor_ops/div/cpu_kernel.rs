use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::ScalarDivKernelOp<F> {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = true;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x / self.scalar
    }
    #[inline(always)]
    fn df(&self, _: &F) -> F {
        F::one() / self.scalar
    }
    #[inline(always)]
    fn const_df(&self) -> F {
        F::one() / self.scalar
    }
}

impl<F: Float> BinaryDerivative<F> for super::BinaryDivKernelOp {
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x / y
    }
    #[inline(always)]
    fn dfdx(&self, _: &F, &y: &F) -> F {
        y.recip()
    }
    #[inline(always)]
    fn dfdy(&self, &x: &F, y: &F) -> F {
        -x / y.powi(2)
    }
}
