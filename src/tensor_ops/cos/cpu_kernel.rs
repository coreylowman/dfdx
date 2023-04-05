use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::CosKernelOp {
    const DF_USES_FX: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.cos()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        -x.sin()
    }
}
