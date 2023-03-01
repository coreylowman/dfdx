use crate::tensor_ops::cpu_kernels::UnaryDerivative;
use num_traits::Float;

impl<F: Float> UnaryDerivative<F> for super::HardswishKernelOp {
    #[inline(always)]
    fn f(&self, &x: &F) -> F {
        if x <= F::from(-3).unwrap() {
            F::zero()
        } else if x >= F::from(3).unwrap() {
            x
        } else {
            x * (x + F::from(3).unwrap()) / F::from(6).unwrap()
        }
    }
    #[inline(always)]
    fn df(&self, &x: &F) -> F {
        if x <= F::from(-3.0).unwrap() {
            F::zero()
        } else if x >= F::from(3.0).unwrap() {
            F::one()
        } else {
            (F::from(2).unwrap() * x + F::from(3).unwrap()) / F::from(6).unwrap()
        }
    }
}
