use super::utils::binary_map;
use crate::gradients::{Merge, Tape};
use crate::prelude::*;

/// Element wise division.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [0.5, 1.0, 3.0]]);
/// let r = div(a, &b); // or `a / &b`
/// assert_eq!(r.data(), &[[1.0, 4.0, 3.0], [-2.0, -2.0, -1.0]]);
/// ```
pub fn div<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs
where
    Lhs: Tensor<Dtype = f32>,
    Rhs: Tensor<Dtype = f32, Array = Lhs::Array>,
    Lhs::Tape: Merge<Rhs::Tape, Output = Lhs::Tape>,
{
    fn dfdy(x: &f32, y: &f32) -> f32 {
        (-x) * y.powi(2).recip()
    }
    binary_map(lhs, rhs, |x, y| x * y.recip(), |_, y| y.recip(), dfdy)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* TapeL: Tape, TapeR: Tape> std::ops::Div<$typename<$($Vs, )* TapeR>> for $typename<$($Vs, )* TapeL>
where
    TapeL: Merge<TapeR, Output = TapeL>
{
    type Output = $typename<$($Vs, )* <TapeL as Merge<TapeR>>::Output>;
    /// Calls [div()] - implements `T<H> / &T<NoneTape>`
    fn div(self, rhs: $typename<$($Vs, )* TapeR>) -> Self::Output {
        div(self, rhs)
    }
}

    };
}

binary_ops_impl!(Tensor0D, []);
binary_ops_impl!(Tensor1D, [N]);
binary_ops_impl!(Tensor2D, [M, N]);
binary_ops_impl!(Tensor3D, [M, N, O]);
binary_ops_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_0d() {
        let a = tensor(2.0);
        let b = tensor(4.0);

        let r = b.trace() / a.clone();
        assert_eq!(r.data(), &2.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
        assert_eq!(gradients.ref_gradient(&b), &0.5);
    }

    #[test]
    fn test_div_1d() {
        let a = tensor([1.0, 2.0, 3.0]);
        let b = tensor([1.0, -1.0, 0.0]);

        let r = b.trace() / a.clone();
        assert_eq!(r.data(), &[1.0, -0.5, 0.0]);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0, 1.0 / 12.0, 0.0]);
        assert_eq!(
            gradients.ref_gradient(&b),
            &[1.0 / 3.0, 1.0 / 6.0, 0.11111112]
        );
    }

    #[test]
    fn test_div_2d() {
        let a = tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = b.trace() / a.clone();
        assert_eq!(
            r.data(),
            &[
                [0.79132426, 2.2505856, 2.506],
                [1.4597031, 0.52524966, 0.046511628]
            ]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [-0.20074181, -2.1961217, -2.7844446],
                [-0.42998204, -0.12488106, -0.009292662]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.25367835, 0.97580016, 1.1111112],
                [0.29456818, 0.2377556, 0.1997922]
            ]
        );
    }
}
