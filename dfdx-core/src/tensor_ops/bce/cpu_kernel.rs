use crate::tensor_ops::cpu_kernels::BinaryDerivative;
use num_traits::Float;

impl<F: Float> BinaryDerivative<F> for super::BCEKernelOp {
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &logit: &F, &prob: &F) -> F {
        logit.max(F::zero()) - logit * prob + (F::one() + (-logit.abs()).exp()).ln()
    }
    #[inline(always)]
    fn dfdx(&self, &logit: &F, &prob: &F) -> F {
        F::one() - prob - (F::one() + logit.exp()).recip()
    }
    #[inline(always)]
    fn dfdy(&self, &logit: &F, _: &F) -> F {
        -logit
    }
}
