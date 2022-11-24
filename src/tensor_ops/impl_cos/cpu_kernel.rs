use super::CosKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for CosKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.cos()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        -x.sin()
    }
}
