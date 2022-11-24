use super::{BinarySubKernelOp, ScalarSubKernelOp};
use crate::tensor_ops::utils::cpu::{BinaryDerivative, UnaryDerivative};

impl UnaryDerivative<f32> for ScalarSubKernelOp<f32> {
    fn f(&self, x: &f32) -> f32 {
        x - self.0
    }
    fn df(&self, _: &f32) -> f32 {
        1.0
    }
}

impl BinaryDerivative<f32> for BinarySubKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x - y
    }
    #[inline(always)]
    fn dfdx(&self, _: &f32, _: &f32) -> f32 {
        1.0
    }
    #[inline(always)]
    fn dfdy(&self, _: &f32, _: &f32) -> f32 {
        -1.0
    }
}
