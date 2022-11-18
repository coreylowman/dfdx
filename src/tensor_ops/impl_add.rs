use crate::{
    arrays::{Dtype, Shape},
    devices::{
        binary_ops,
        device::{BinaryKernel, HasErr},
        Device,
    },
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::try_binary_op;

/// Element wise addition.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = Tensor2D::ones();
/// let r = add(a, b); // or `a + b`
/// assert_eq!(r.data(), &[[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
pub trait TryAdd<Rhs = Self>: HasErr {
    fn try_add(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryAdd<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<binary_ops::Add, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_add(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    std::ops::Add<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    Self: TryAdd<Tensor<S, E, D, RhsTape>>,
{
    type Output = Self;
    fn add(self, rhs: Tensor<S, E, D, RhsTape>) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

// /// Element wise addition.
// ///
// /// Example:
// /// ```rust
// /// # use dfdx::prelude::*;
// /// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
// /// let b = Tensor2D::ones();
// /// let r = add(a, b); // or `a + b`
// /// assert_eq!(r.data(), &[[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
// /// ```
// pub fn add<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs
// where
//     Lhs: Tensor<Dtype = f32>,
//     Rhs: Tensor<Dtype = f32, Array = Lhs::Array>,
//     Lhs::Tape: Merge<Rhs::Tape>,
// {
//     binary_map(lhs, rhs, |x, y| x + y, |_, _| 1.0, |_, _| 1.0)
// }

// macro_rules! binary_ops_impl {
//     ($typename:ident, [$($Vs:tt),*]) => {
// impl<$(const $Vs: usize, )* TapeL: Tape, TapeR: Tape> std::ops::Add<$typename<$($Vs, )* TapeR>> for $typename<$($Vs, )* TapeL>
// where
//     TapeL: Merge<TapeR>
// {
//     type Output = $typename<$($Vs, )* TapeL>;
//     /// Calls [add()] - implements `T<L> + T<R>`
//     fn add(self, rhs: $typename<$($Vs, )* TapeR>) -> Self::Output {
//         add(self, rhs)
//     }
// }
//     };
// }

// binary_ops_impl!(Tensor0D, []);
// binary_ops_impl!(Tensor1D, [N]);
// binary_ops_impl!(Tensor2D, [M, N]);
// binary_ops_impl!(Tensor3D, [M, N, O]);
// binary_ops_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use crate::devices::AsArray;
    use crate::tensor::TensorSugar;
    use crate::tensor_ops::impl_backward::TryBackward;
    use crate::tests::build_test_device;

    #[test]
    fn test_add_0d() {
        let dev = build_test_device!();
        let a = dev.tensor(1.0);
        let b = dev.tensor(1.0);

        let r = a.trace() + b.clone();
        assert_eq!(r.as_array(), 2.0);
        let g = r.backward();
        assert_eq!(g.get(&a).as_array(), 1.0);
        assert_eq!(g.get(&b).as_array(), 1.0);
    }

    // #[test]
    // fn test_add_1d() {
    //     let dev = build_test_device!();
    //     let a = dev.tensor([1.0, 2.0, 3.0]);
    //     let b = dev.tensor([1.0, -1.0, 0.0]);

    //     let r = a.trace() + b.clone();
    //     assert_eq!(r.as_array(), [2.0, 1.0, 3.0]);
    //     let gradients = backward(r.mean());
    //     assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
    //     assert_eq!(gradients.ref_gradient(&b), &[1.0 / 3.0; 3]);
    // }

    // #[test]
    // fn test_add_2d() {
    //     let dev = build_test_device!();
    //     let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
    //     let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

    //     let r = a.trace() + b.clone();
    //     assert_eq!(
    //         r.as_array(),
    //         [[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]
    //     );
    //     let gradients = backward(r.mean());
    //     assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 6.0; 3]; 2]);
    //     assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);
    // }
}
