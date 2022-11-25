use super::utils::move_tape_and_add_backward_op;
use crate::devices::{AddAccum, DeviceReduce};
use crate::gradients::Tape;
use crate::prelude::*;

/// Sum values along axes `Axes` of `T`.
///
/// **Pytorch equivalent**: `t.sum(Axes)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor0D = t.sum();
/// ```
///
/// Reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor2D<3, 4> = t.sum();
/// ```
///
/// Reducing multiple axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor1D<4> = t.sum();
/// ```
pub fn sum<T: Reduce<Axes>, Axes>(t: T) -> T::Reduced {
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::DeviceR::reduce_into_no_reset::<AddAccum>(result.mut_data(), t.data());
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::DeviceR::broadcast_into_no_reset::<AddAccum>(t_grad, result_grad);
    })
}

macro_rules! sum_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [sum()].
    pub fn sum<T, Axes>(self) -> T where Self: ReduceTo<T, Axes> {
        sum(self)
    }
}
    };
}

sum_axis_impl!(Tensor0D, []);
sum_axis_impl!(Tensor1D, [M]);
sum_axis_impl!(Tensor2D, [M, N]);
sum_axis_impl!(Tensor3D, [M, N, O]);
sum_axis_impl!(Tensor4D, [M, N, O, P]);
sum_axis_impl!(Tensor5D, [M, N, O, P, Q]);
sum_axis_impl!(Tensor6D, [M, N, O, P, Q, R]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::thread_rng;

    #[test]
    fn test_sum_1d() {
        let t: Tensor1D<3> = tensor([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().sum();
        assert_eq!(r.data(), &6.0);
        // NOTE: .exp() to make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[403.4288; 3]);
    }

    #[test]
    fn test_valids_sum_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().sum();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().sum();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().sum();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().sum();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().sum();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().sum();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().sum();

        let _: Tensor0D = Tensor4D::<9, 7, 5, 3>::zeros().sum();
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().sum::<_, Axis<0>>();
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
        let r = t.trace().sum::<_, Axis<1>>();
        assert_eq!(r.data(), &[6.0, -4.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[201.7144; 3], [0.00915782; 3]]
        );
    }

    #[test]
    fn test_sum_axes_3d_to_1d() {
        let mut rng = thread_rng();
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<3, _> = t.trace().sum::<_, Axes2<0, 2>>();
        let r2: Tensor1D<3, _> = t.trace().sum::<_, Axis<0>>().sum::<_, Axis<1>>();
        assert_close(r.data(), r2.data());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(g.ref_gradient(&t), g2.ref_gradient(&t));
    }
}
