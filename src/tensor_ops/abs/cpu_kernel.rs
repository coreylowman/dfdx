use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::AbsKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.abs()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x == &0.0 {
            0.0
        } else {
            x.signum()
        }
    }
}
