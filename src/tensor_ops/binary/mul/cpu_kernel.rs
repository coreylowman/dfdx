use crate::tensor_ops::cpu_kernels::{BinaryDerivative, UnaryDerivative};

impl UnaryDerivative<f32> for super::ScalarMulKernelOp<f32> {
    fn f(&self, x: &f32) -> f32 {
        x * self.scalar
    }
    fn df(&self, _: &f32) -> f32 {
        self.scalar
    }
}

impl BinaryDerivative<f32> for super::BinaryMulKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32, y: &f32) -> f32 {
        x * y
    }
    #[inline(always)]
    fn dfdx(&self, _x: &f32, y: &f32) -> f32 {
        *y
    }
    #[inline(always)]
    fn dfdy(&self, x: &f32, _y: &f32) -> f32 {
        *x
    }
}
