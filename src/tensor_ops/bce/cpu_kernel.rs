use crate::tensor_ops::cpu_kernels::BinaryDerivative;

impl BinaryDerivative<f32> for super::BCEKernelOp {
    #[inline(always)]
    fn f(&self, logit: &f32, prob: &f32) -> f32 {
        logit.max(0.0) - logit * prob + (1.0 + (-logit.abs()).exp()).ln()
    }
    #[inline(always)]
    fn dfdx(&self, logit: &f32, prob: &f32) -> f32 {
        1.0 - prob - (1.0 + logit.exp()).recip()
    }
    #[inline(always)]
    fn dfdy(&self, logit: &f32, _: &f32) -> f32 {
        -logit
    }
}
