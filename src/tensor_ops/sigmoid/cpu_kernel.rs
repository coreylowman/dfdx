use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for super::SigmoidKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        let fx = 1.0 / (1.0 + (-x).exp());
        fx * (1.0 - fx)
    }
}
