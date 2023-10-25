use crate::tensor_ops::cpu_kernels::BinaryDerivative;

impl<F: num_traits::Float> BinaryDerivative<F> for super::MaximumKernelOp {
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x.max(y)
    }
    #[inline(always)]
    fn dfdx(&self, x: &F, y: &F) -> F {
        if x > y {
            F::one()
        } else if x < y {
            F::zero()
        } else {
            F::from(0.5).unwrap()
        }
    }
    #[inline(always)]
    fn dfdy(&self, x: &F, y: &F) -> F {
        if y > x {
            F::one()
        } else if y < x {
            F::zero()
        } else {
            F::from(0.5).unwrap()
        }
    }
}
