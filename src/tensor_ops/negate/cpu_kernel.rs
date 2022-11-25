use super::NegateKernelOp;
use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl UnaryDerivative<f32> for NegateKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        -x
    }
    #[inline(always)]
    fn df(&self, _: &f32) -> f32 {
        -1.0
    }
}
