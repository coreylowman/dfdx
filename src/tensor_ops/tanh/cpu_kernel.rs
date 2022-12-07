use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::TanhKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.tanh()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
}
