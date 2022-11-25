use super::LnKernelOp;
use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for LnKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.ln()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        1.0 / x
    }
}
