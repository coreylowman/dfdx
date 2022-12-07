use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::PowKernelOp<i32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powi(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        self.0 as f32 * x.powi(self.0 - 1)
    }
}

impl UnaryDerivative<f32> for super::PowKernelOp<f32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powf(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        self.0 * x.powf(self.0 - 1.0)
    }
}
