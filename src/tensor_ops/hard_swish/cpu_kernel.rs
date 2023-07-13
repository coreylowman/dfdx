use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::HardSwishKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        x * (x + F::from(3.0).unwrap())
            .max(F::zero())
            .min(F::from(6.0).unwrap())
            / F::from(6.0).unwrap()
    }
    #[inline(always)]
    fn df(&self, &x: &F) -> F {
        if x > F::from(-3.0).unwrap() {
            (x + x + F::from(3.0).unwrap()) / F::from(6.0).unwrap()
        } else {
            F::zero()
        }
    }
}
