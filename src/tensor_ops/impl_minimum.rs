use super::utils::binary_map;
use crate::gradients::{Merge, Tape};
use crate::prelude::*;

/// Element wise minimum.
///
/// **Pytorch equivalent**: `torch.minimum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.minimum(&b);
/// assert_eq!(r.data(), &[[1.0, 0.5, 1.0], [-2.0, -2.0, -3.5]]);
pub fn minimum<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs
where
    Lhs: Tensor<Dtype = f32>,
    Rhs: Tensor<Dtype = f32, Array = Lhs::Array>,
    Lhs::Tape: Merge<Rhs::Tape>,
{
    fn f(x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    fn dfdx(x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }

    fn dfdy(x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* TapeL: Tape> $typename<$($Vs, )* TapeL> {
    /// Calls [minimum()] on `self`.
    pub fn minimum<TapeR: Tape>(self, other: $typename<$($Vs, )* TapeR>) -> $typename<$($Vs, )* TapeL>
    where
        TapeL: Merge<TapeR>
    {
        minimum(self, other)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimum() {
        let a = tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = minimum(a.trace(), b.clone());
        assert_eq!(result.data(), &[[-1., 0., -1.], [3., -4., -5.]]);

        let g = backward(result.sum());
        assert_eq!(g.ref_gradient(&a), &[[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
        assert_eq!(g.ref_gradient(&b), &[[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
    }
}
