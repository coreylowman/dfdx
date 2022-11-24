use super::MinimumKernelOp;
use crate::tensor_ops::utils::cpu::BinaryDerivative;

impl BinaryDerivative<f32> for MinimumKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    #[inline(always)]
    fn dfdx(&self, x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }

    #[inline(always)]
    fn dfdy(&self, x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
}