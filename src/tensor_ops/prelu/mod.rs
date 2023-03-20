mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use core::fmt::Debug;

use crate::{shapes::*, tensor::*};

use super::Device;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct PReLUKernelOp;

/// [Parametric Rectified Linear Unit (PReLU)](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html). `max(0, t) + a*min(0, t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let a = dev.tensor(0.05);
/// let r = prelu(t, a);
/// assert_eq!(r.array(), [-0.05, 0.0, 1.0, 2.0]);
/// ```

pub fn prelu<
    S: Shape,
    E: Dtype,
    D: Device<E> + PReLUKernel<Tensor<S, E, D>, Tensor<(), E, D>, Output = Tensor<S, E, D>, Elem = E>,
    T: Tape<E, D> + Merge<R>,
    R: Default,
>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<(), E, D, R>,
) -> Tensor<S, E, D, T> {
    lhs.prelu(rhs)
}

pub trait PReLUKernel<L, R>: DeviceStorage {
    type Output: HasErr;
    type Elem: Dtype;

    fn forward(&self, lhs: &L, rhs: &R) -> Result<Self::Output, <Self::Output as HasErr>::Err>;

    fn backward(
        &self,
        lhs: &L,
        lhs_grad: &mut <Self as storage_traits::DeviceStorage>::Vec<Self::Elem>,
        rhs: &R,
        rhs_grad: &mut <Self as storage_traits::DeviceStorage>::Vec<Self::Elem>,
        grad: &<Self as storage_traits::DeviceStorage>::Vec<Self::Elem>,
    ) -> Result<(), <Self::Output as HasErr>::Err>;
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    pub fn prelu<R: Default>(self, other: Tensor<(), E, D, R>) -> Self
    where
        D: PReLUKernel<Tensor<S, E, D>, Tensor<(), E, D>, Output = Tensor<S, E, D>, Elem = E>,
        T: Merge<R>,
    {
        self.try_prelu(other).unwrap()
    }

    pub fn try_prelu<R: Default>(
        self,
        other: Tensor<(), E, D, R>,
    ) -> Result<Self, <Self as HasErr>::Err>
    where
        D: PReLUKernel<Tensor<S, E, D>, Tensor<(), E, D>, Output = Tensor<S, E, D>, Elem = E>,
        T: Merge<R>,
    {
        let device = D::default();

        let (lhs, lt) = self.split_tape();
        let (rhs, rt) = other.split_tape();
        let out = PReLUKernel::forward(&device, &lhs, &rhs)?;

        let mut tape = lt.merge(rt);

        let phantom_out = out.clone();
        tape.try_alloc_grad(&lhs)?;
        tape.try_alloc_grad(&rhs)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
            PReLUKernel::backward(&device, &lhs, grad_lhs, &rhs, grad_rhs, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_prelu() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y: Tensor<_, TestDtype, _> = dev.tensor(0.05);
        let r = x.leaky_trace().prelu(y.clone());
        assert_eq!(r.array(), [-0.1, -0.05, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .prelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[0.00904837, 0.00951229, 0.01, 0.54365635, 1.4778112],
        );
        // TODO: confirm this
        assert_close(&g.get(&y).array(), &-0.11043618);
    }
}
