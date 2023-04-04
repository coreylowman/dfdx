use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::{clamp, Float};

impl<F: Float + PartialOrd> UnaryDerivative<F> for super::ClampKernelOp<F> {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        clamp(x, self.min, self.max)
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if (self.min..=self.max).contains(x) {
            F::one()
        } else {
            F::zero()
        }
    }
}
