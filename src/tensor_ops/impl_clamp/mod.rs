mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr, UnaryKernel},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

/// Clamp all elements between the provided min and max values.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, -0.5, 0.0, 0.5, 1.0]);
/// let r = t.clamp(-0.5, 0.5);
/// assert_eq!(r.data(), &[-0.5, -0.5, 0.0, 0.5, 0.5]);
/// ```
pub trait TryClamp<E: Dtype>: HasErr {
    fn clamp(self, min: E, max: E) -> Self {
        self.try_clamp(min, max).unwrap()
    }
    fn try_clamp(self, min: E, max: E) -> Result<Self, Self::Err>;
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ClampKernelOp<E> {
    pub min: E,
    pub max: E,
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryClamp<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<ClampKernelOp<E>, S, S, E>,
{
    fn try_clamp(self, min: E, max: E) -> Result<Self, Self::Err> {
        try_unary_op(ClampKernelOp { min, max }, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        devices::AsArray,
        tensor::{Tensor2D, TensorSugar},
        tensor_ops::{impl_backward::TryBackward, impl_mean::MeanTo, map::TryExp},
        tests::build_test_device,
    };

    use super::*;

    #[test]
    fn test_clamp() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 1.1]]);
        let r = t.trace().clamp(-1.0, 1.0);
        assert_eq!(r.as_array(), [[-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.06131324, 0.16666667, 0.45304698], [0.0; 3]]
        );
    }
}
