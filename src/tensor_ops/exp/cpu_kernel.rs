use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::ExpKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.exp()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        x.exp()
    }
}
