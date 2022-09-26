use super::utils::move_tape_and_add_backward_op;
use crate::devices::broadcast_reduce::{AddAccum, DeviceReduce};
use crate::prelude::*;

/// Sum the values along dimension `I` of `T`.
///
/// **Pytorch equivalent**: `t.sum(I)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = TensorCreator::zeros();
/// let a: Tensor1D<3> = t.clone().sum_axis::<0>();
/// let b: Tensor1D<2> = t.sum_axis::<-1>();
/// ```
pub fn sum_axis<T: Reduce<Axis<I>>, const I: isize>(t: T) -> T::Reduced {
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::DeviceR::reduce_into::<AddAccum>(result.mut_data(), t.data());
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::DeviceR::broadcast_into::<AddAccum>(t_grad, result_grad);
    })
}

macro_rules! sum_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [sum_axis()] on `self`.
    pub fn sum_axis<const I: isize>(self) -> <Self as Reduce<Axis<I>>>::Reduced
    where
        Self: Reduce<Axis<I>>
    {
        sum_axis::<Self, I>(self)
    }
}
    };
}

sum_axis_impl!(Tensor0D, []);
sum_axis_impl!(Tensor1D, [M]);
sum_axis_impl!(Tensor2D, [M, N]);
sum_axis_impl!(Tensor3D, [M, N, O]);
sum_axis_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valids_sum_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().sum_axis::<-1>();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().sum_axis::<0>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().sum_axis::<-1>();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<0>();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<1>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<-1>();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<0>();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<1>();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<2>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<-1>();
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().sum_axis::<0>();
        assert_eq!(r.data(), &[-1.0, 6.0, -3.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.12262648, 134.47627, 0.01659569]; 2]
        );
    }

    #[test]
    fn test_sum_axis_1_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().sum_axis::<-1>();
        assert_eq!(r.data(), &[6.0, -4.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[201.7144; 3], [0.00915782; 3]]
        );
    }
}
