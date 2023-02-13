use crate::tensor_ops::cpu_kernels::BinaryDerivative;

use num_traits::Float;

impl<F: Float> BinaryDerivative<F> for super::HuberErrorKernelOp<F> {
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        let half = F::from(0.5).unwrap();
        if (x - y).abs() < self.delta {
            (x - y).powi(2) * half
        } else {
            (x - y).abs() * self.delta - half * self.delta * self.delta
        }
    }

    #[inline(always)]
    fn dfdx(&self, &x: &F, &y: &F) -> F {
        if (x - y) == F::zero() {
            F::zero()
        } else if (x - y).abs() < self.delta {
            x - y
        } else {
            (x - y).signum() * self.delta
        }
    }

    #[inline(always)]
    fn dfdy(&self, &x: &F, &y: &F) -> F {
        if (x - y) == F::zero() {
            F::zero()
        } else if (x - y).abs() < self.delta {
            y - x
        } else {
            (y - x).signum() * self.delta
        }
    }
}
