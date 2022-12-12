use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::ReLUKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.max(0.0)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        if x > &0.0 {
            1.0
        } else {
            0.0
        }
    }
}
