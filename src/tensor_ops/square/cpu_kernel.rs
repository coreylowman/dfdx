use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::SquareKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.powi(2)
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        2.0 * x
    }
}
