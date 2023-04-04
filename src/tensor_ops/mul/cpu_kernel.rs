use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};

use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::ScalarMulKernelOp<F> {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = true;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x * self.scalar
    }
    #[inline(always)]
    fn df(&self, _: &F) -> F {
        self.scalar
    }
    fn const_df(&self) -> F {
        self.scalar
    }
}

impl<F: Float> BinaryDerivative<F> for super::BinaryMulKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x * y
    }
    #[inline(always)]
    fn dfdx(&self, _x: &F, &y: &F) -> F {
        y
    }
    #[inline(always)]
    fn dfdy(&self, &x: &F, _y: &F) -> F {
        x
    }
}
