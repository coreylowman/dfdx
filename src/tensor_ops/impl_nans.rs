use crate::{
    arrays::{Dtype, Shape},
    devices::{
        device::{HasErr, UnaryKernel},
        unary_ops, Device,
    },
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

/// Replaces any [std::f32::NAN] with `value`.
///
/// **Pytorch equivalent**: `t.nan_to_num(value)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor1D<4> = tensor([1.0, f32::NAN, f32::NAN, 4.0]);
/// let r = t.nans_to(0.0);
/// assert_eq!(r.data(), &[1.0, 0.0, 0.0, 4.0]);
/// ```
pub trait TryNansTo<E: Dtype>: HasErr {
    fn nans_to(self, value: E) -> Self {
        self.try_nans_to(value).unwrap()
    }
    fn try_nans_to(self, value: E) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryNansTo<E> for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::NansTo<E>, S, S, E>,
{
    fn try_nans_to(self, value: E) -> Result<Self, Self::Err> {
        try_unary_op(unary_ops::NansTo(value), self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        devices::AsArray,
        tensor::TensorSugar,
        tensor_ops::{impl_backward::TryBackward, impl_mean::MeanTo, map::TryExp},
        tests::build_test_device,
    };

    #[test]
    fn test_nans_1d() {
        let dev = build_test_device!();
        let t = dev.tensor([1.0, f32::NAN, -f32::NAN, 4.0]);
        let r = t.trace().nans_to(0.0);
        assert_eq!(r.as_array(), [1.0, 0.0, 0.0, 4.0]);
        // NOTE: .exp() so we cover case where nans_to() needs to use result grad
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).as_array(), [0.67957044, 0.0, 0.0, 13.649537]);
    }
}
