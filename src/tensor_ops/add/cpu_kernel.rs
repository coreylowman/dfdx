use super::{BinaryAddKernelOp, ScalarAddKernelOp};
use crate::tensor_ops::utils::cpu::{BinaryDerivative, UnaryDerivative};

impl BinaryDerivative<f32> for BinaryAddKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dfdx(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
    #[inline(always)]
    fn dfdy(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
}

impl UnaryDerivative<f32> for ScalarAddKernelOp<f32> {
    fn f(&self, x: &f32) -> f32 {
        x + self.0
    }
    fn df(&self, _: &f32) -> f32 {
        1.0
    }
}