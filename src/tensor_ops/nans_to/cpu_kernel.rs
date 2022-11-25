use super::NansToKernelOp;
use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for NansToKernelOp<f32> {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        if x.is_nan() {
            self.0
        } else {
            *x
        }
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x.is_nan() {
            0.0
        } else {
            1.0
        }
    }
}
