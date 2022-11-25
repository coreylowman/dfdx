use super::utils::move_tape_and_add_backward_op;
use crate::devices::{Device, DeviceReduce, EqAccum, MinAccum, MulAccum};
use crate::gradients::Tape;
use crate::prelude::*;

/// Reduces `Axes` of the tensor by gathering the minimum value from the axes.
///
/// **Pytorch equivalent**: `t.amin(Axes)`
///
/// **NOTE** This evenly distributes gradients between all equal minimum values, instead
/// of only exactly 1 value.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = t.min();
/// assert_eq!(r.data(), &[1.0, -3.0]);
/// ```
///
/// Reducing 2 axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor0D = t.min();
/// assert_eq!(r.data(), &-3.0);
/// ```
pub fn min<T: Reduce<Axes>, Axes>(mut t: T) -> T::Reduced {
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::DeviceR::reduce_into::<MinAccum>(result.mut_data(), t.data());

    // store derivative in t
    T::DeviceR::broadcast_into_no_reset::<EqAccum>(t.mut_data(), result.data());

    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::DeviceR::broadcast_into_no_reset::<MulAccum>(t.mut_data(), result_grad);
        T::Device::add(t_grad, t.data());
    })
}

macro_rules! min_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [min()]
    pub fn min<T, Axes>(self) -> T where Self: ReduceTo<T, Axes> {
        min(self)
    }
}
    };
}

min_axis_impl!(Tensor0D, []);
min_axis_impl!(Tensor1D, [M]);
min_axis_impl!(Tensor2D, [M, N]);
min_axis_impl!(Tensor3D, [M, N, O]);
min_axis_impl!(Tensor4D, [M, N, O, P]);
min_axis_impl!(Tensor5D, [M, N, O, P, Q]);
min_axis_impl!(Tensor6D, [M, N, O, P, Q, R]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::thread_rng;

    #[test]
    fn test_valids_min_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().min();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().min();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().min();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().min();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().min();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().min();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().min();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().min();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().min();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().min();
    }

    #[test]
    fn test_min_axis_0_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().min::<_, Axis<0>>();
        assert_eq!(r.data(), &[1.0, -2.0, 2.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.ref_gradient(&t),
            &[[0.90609396, 0.0, 2.463019], [0.0, 0.04511176, 2.463019]]
        );
    }

    #[test]
    fn test_min_axis_1_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().min::<_, Axis<1>>();
        assert_eq!(r.data(), &[1.0, -2.0]);
        let g = r.sum().backward();
        assert_eq!(g.ref_gradient(&t), &[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_min_axes_3d_to_1d() {
        let mut rng = thread_rng();
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<4, _> = t.trace().min::<_, Axes2<0, 1>>();
        let r2: Tensor1D<4, _> = t.trace().min::<_, Axis<0>>().min::<_, Axis<0>>();
        assert_close(r.data(), r2.data());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(g.ref_gradient(&t), g2.ref_gradient(&t));
    }
}
