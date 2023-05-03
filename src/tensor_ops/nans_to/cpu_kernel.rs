use crate::{shapes::Dtype, tensor_ops::cpu_kernels::UnaryDerivative};

impl<F: Dtype + num_traits::Float> UnaryDerivative<F> for super::NansToKernelOp<F> {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        if x.is_nan() {
            self.0
        } else {
            *x
        }
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        if x.is_nan() {
            F::zero()
        } else {
            F::one()
        }
    }
}
