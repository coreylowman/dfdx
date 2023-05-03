use crate::{shapes::Dtype, tensor_ops::cpu_kernels::UnaryDerivative};

impl<F: num_traits::Float + Dtype> UnaryDerivative<F> for super::PowiKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.powi(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        F::from(self.0).unwrap() * x.powi(self.0 - 1)
    }
}

impl<F: num_traits::Float + Dtype> UnaryDerivative<F> for super::PowfKernelOp<F> {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.powf(self.0)
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        self.0 * x.powf(self.0 - F::one())
    }
}
