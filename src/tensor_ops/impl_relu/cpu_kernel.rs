use super::ReLUKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for ReLUKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.max(0.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x > &0.0 {
            1.0
        } else {
            0.0
        }
    }
}
