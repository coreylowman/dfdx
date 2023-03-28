extern crate alloc;
use alloc::{sync::Arc, vec::Vec};
use num_traits::Float;

use crate::prelude::{
    cpu_kernels::BinaryDerivative, unique_id, Cpu, Dtype, HasErr, Merge, Shape, Tensor,
};

use super::{PReLUKernel, PReLUKernelOp};

impl<E: Float> BinaryDerivative<E> for PReLUKernelOp {
    fn f(&self, x: &E, y: &E) -> E {
        let zero = E::from(0.0).unwrap();
        x.max(zero) + *y * x.min(zero)
    }

    fn dfdx(&self, x: &E, y: &E) -> E {
        let zero = E::from(0.0).unwrap();
        let one = E::from(1.0).unwrap();
        if x >= &zero {
            one
        } else {
            *y
        }
    }

    fn dfdy(&self, x: &E, _y: &E) -> E {
        let zero = E::from(0.0).unwrap();
        if x >= &zero {
            zero
        } else {
            *x
        }
    }
}

impl<S: Shape, E: Dtype + Float> PReLUKernel<Tensor<S, E, Cpu>, Tensor<(), E, Cpu>> for Cpu {
    type Output = Tensor<S, E, Cpu>;

    type Elem = E;

    fn forward(
        &self,
        lhs: &Tensor<S, E, Cpu>,
        rhs: &Tensor<(), E, Cpu>,
    ) -> Result<Self::Output, <Self::Output as HasErr>::Err> {
        let p = PReLUKernelOp;
        let rv = rhs.data.get(0).unwrap();
        let new_data = lhs
            .data
            .iter()
            .copied()
            .map(|x| p.f(&x, rv))
            .collect::<Vec<_>>();

        let t = Tensor {
            id: unique_id(),
            data: Arc::new(new_data),
            shape: lhs.shape,
            strides: lhs.strides,
            device: lhs.device.clone(),
            tape: lhs.tape.merge(rhs.tape),
        };
        Ok(t)
    }

    fn backward(
        &self,
        lhs: &Tensor<S, E, Cpu>,
        lhs_grad: &mut <Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
        rhs: &Tensor<(), E, Cpu>,
        rhs_grad: &mut <Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
        grad: &<Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
    ) -> Result<(), <Self::Output as HasErr>::Err> {
        let op = PReLUKernelOp;
        let rg = rhs_grad.get_mut(0).expect("A must be a single value"); // TODO not expect
        let r = *rhs.as_vec().get(0).expect("A must be a single value");

        // Should I do this?
        let scale = E::from_f32(1.0 / lhs_grad.len() as f32).unwrap();
        lhs_grad
            .iter_mut()
            .zip(lhs.as_vec().iter())
            .zip(grad)
            .for_each(|((lg, l), g)| {
                *(lg) += op.dfdx(l, &r) * *g;
                *(rg) += op.dfdy(l, &r) * *g * scale;
            });
        Ok(())
    }
}
