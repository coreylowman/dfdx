mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

/// [P Rectified Linear Unit (PReLU)](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html). `max(0, t) + a*min(0, t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.prelu(0.05);
/// assert_eq!(r.array(), [-0.05, 0.0, 1.0, 2.0]);
/// ```
pub fn prelu<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    a: E,
) -> Tensor<S, E, D, T> 
where Tensor<S, E, D, T>: PReLUDev<E, D>{
    t.prelu(a)
}

pub trait PReLUDev<E: Dtype, D: DeviceStorage> : Sized{
    /// See [prelu]
    fn prelu(self, a: E) -> Self {
        self.try_prelu(a).unwrap()
    }
    /// See [prelu]
    fn try_prelu(self, a: E) -> Result<Self, D::Err>;
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::{*, prelu::PReLUDev}, tests::*};

    #[test]
    fn test_prelu() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().prelu(0.05);
        assert_eq!(r.array(), [-0.1, -0.05, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .relu() uses the result's gradient
        let g = r.exp().mean().backward();
        // TODO
        // assert_close(&g.get(&x).array(), &[0.0, 0.0, 0.0, 0.54365635, 1.4778112]);
    }
}
