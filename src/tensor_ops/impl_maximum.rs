use super::utils::binary_map;
use crate::prelude::*;

/// Element wise maximum.
///
/// **Pytorch equivalent**: `torch.maximum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.maximum(&b);
/// assert_eq!(r.data(), &[[1.0, 2.0, 3.0], [-1.0, 2.0, -3.0]]);
pub fn maximum<T: Tensor<Dtype = f32>>(lhs: T, rhs: &T::NoTape) -> T {
    fn f(x: &f32, y: &f32) -> f32 {
        x.max(*y)
    }
    fn dfdx(x: &f32, y: &f32) -> f32 {
        if x > y {
            1.0
        } else if x < y {
            0.0
        } else {
            0.5
        }
    }

    fn dfdy(x: &f32, y: &f32) -> f32 {
        if y > x {
            1.0
        } else if y < x {
            0.0
        } else {
            0.5
        }
    }
    binary_map(lhs, rhs, f, dfdx, dfdy)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [maximum()] on `self`.
    pub fn maximum(self, other: &<Self as Tensor>::NoTape) -> Self {
        maximum(self, other)
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
    fn test_maximum() {
        let a = tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = maximum(a.trace(), &b);
        assert_eq!(result.data(), &[[0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]);

        let g = result.sum::<_, AllAxes>().backward();
        assert_eq!(g.ref_gradient(&a), &[[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
        assert_eq!(g.ref_gradient(&b), &[[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
    }
}
