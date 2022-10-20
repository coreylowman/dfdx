use super::utils::{binary_map, BinaryOpTyping};
use crate::gradients::{Merge, Tape};
use crate::prelude::*;

/// Element wise multiplication.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = mul(a, &b); // or `a * &b`
/// assert_eq!(r.data(), &[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// ```
pub fn mul<Lhs, Rhs, Out>(lhs: Lhs, rhs: Rhs) -> Out
where
    Lhs: Tensor<Dtype = f32> + BinaryOpTyping<Rhs, Out = Out>,
    Rhs: Tensor<Dtype = f32, Array = Lhs::Array>,
    Out: Tensor<Dtype = f32, Array = Lhs::Array, Tape = <Lhs::Tape as Merge<Rhs::Tape>>::Output>,
    Lhs::Tape: Merge<Rhs::Tape>,
{
    binary_map(lhs, rhs, |x, y| x * y, |_, y| *y, |x, _| *x)
}

macro_rules! binary_ops_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* TapeL: Tape, TapeR: Tape> std::ops::Mul<$typename<$($Vs, )* TapeR>> for $typename<$($Vs, )* TapeL>
where
    TapeL: Merge<TapeR>,
{
    type Output = $typename<$($Vs, )* <TapeL as Merge<TapeR>>::Output>;
    /// Calls [add()] - implements `T<H> + &T<I>`
    fn mul(self, rhs: $typename<$($Vs, )* TapeR>) -> Self::Output {
        mul(self, rhs)
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
    fn test_mul_0d() {
        let a = tensor(2.0);
        let b = tensor(3.0);

        let r = a.trace() * b.clone();
        assert_eq!(r.data(), &6.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &3.0);
        assert_eq!(gradients.ref_gradient(&b), &2.0);
    }

    #[test]
    fn test_mul_1d() {
        let a = tensor([1.0, 2.0, 3.0]);
        let b = tensor([1.0, -1.0, 0.0]);

        let r = a.trace() * b.clone();
        assert_eq!(r.data(), &[1.0, -2.0, 0.0]);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0, -1.0 / 3.0, 0.0]);
        assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn test_mul_2d() {
        let a = tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.trace() * b.clone();
        assert_eq!(
            r.data(),
            &[
                [0.3415743, 0.06565552, 0.056385003],
                [0.46729425, 0.2581082, 0.03236696]
            ]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.08665001, 0.06406667, 0.06265],
                [0.13765001, 0.06136667, 0.006466667]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.109500006, 0.028466668, 0.025000002],
                [0.0943, 0.11683333, 0.13903335]
            ]
        );
    }
}
