extern crate alloc;

use num_traits::Float;

use crate::prelude::cpu_kernels::{BinaryDerivative, UnaryDerivative};

use super::{LeakyReLUKernelOp, PReLUKernelOp};

impl<E: Float> BinaryDerivative<E> for PReLUKernelOp {
    fn f(&self, x: &E, y: &E) -> E {
        let zero = E::zero();
        x.max(zero) + *y * x.min(zero)
    }

    fn dfdx(&self, x: &E, y: &E) -> E {
        let zero = E::zero();
        let one = E::one();
        if x >= &zero {
            one
        } else {
            *y
        }
    }

    fn dfdy(&self, x: &E, _y: &E) -> E {
        let zero = E::zero();
        if x >= &zero {
            zero
        } else {
            *x
        }
    }
}

impl<E: Float> UnaryDerivative<E> for LeakyReLUKernelOp<E> {
    fn f(&self, x: &E) -> E {
        let zero = E::zero();
        x.max(zero) + self.slope * x.min(zero)
    }

    fn df(&self, x: &E) -> E {
        let zero = E::zero();
        let one = E::one();
        if x >= &zero {
            one
        } else {
            self.slope
        }
    }
}
