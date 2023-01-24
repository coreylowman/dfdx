use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::SqrtKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.sqrt()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        0.5 / x.sqrt()
    }
}
