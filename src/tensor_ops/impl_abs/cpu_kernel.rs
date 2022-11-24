use super::AbsKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for AbsKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.abs()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x == &0.0 {
            0.0
        } else {
            x.signum()
        }
    }
}
