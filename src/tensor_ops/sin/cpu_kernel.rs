use super::SinKernelOp;
use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for SinKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.sin()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        x.cos()
    }
}
