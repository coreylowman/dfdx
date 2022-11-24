use super::SquareKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for SquareKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powi(2)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        2.0 * x
    }
}
