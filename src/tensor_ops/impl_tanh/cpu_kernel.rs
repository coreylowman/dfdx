use super::TanhKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for TanhKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.tanh()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
}
