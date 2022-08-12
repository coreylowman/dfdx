use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Reduces dimension `I` of the tensor by gathering the maximum value from that dimension.
///
/// **Pytorch equivalent**: `t.amax(I)`
///
/// **NOTE** This evenly distributes gradients between all equal maximum values, instead
/// of only exactly 1 value.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = t.max_axis::<-1>();
/// assert_eq!(r.data(), &[3.0, -1.0]);
/// ```
pub fn max_axis<T: Reduce1<I>, const I: isize>(mut t: T) -> T::Reduced {
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::DeviceR::reduce_into(t.data(), result.mut_data(), f32::max);

    // store derivative in t
    T::DeviceR::foreach_br(t.mut_data(), result.data(), &mut |l, r| {
        *l = if l == r { 1.0 } else { 0.0 }
    });

    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);

        T::DeviceR::foreach_br(t.mut_data(), result_grad, &mut |d, r| {
            *d *= r;
        });
        T::Device::foreach_mr(t_grad, t.data(), &mut |g, dr| {
            *g += dr;
        });
    })
}

macro_rules! max_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [max_axis()] on `self`.
    pub fn max_axis<const I: isize>(self) -> <Self as Reduce1<I>>::Reduced
    where
        Self: Reduce1<I>,
    {
        max_axis::<Self, I>(self)
    }
}
    };
}

max_axis_impl!(Tensor0D, []);
max_axis_impl!(Tensor1D, [M]);
max_axis_impl!(Tensor2D, [M, N]);
max_axis_impl!(Tensor3D, [M, N, O]);
max_axis_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valids_max_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().max_axis::<0>();
        let _: Tensor0D = Tensor1D::<5>::zeros().max_axis::<-1>();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().max_axis::<0>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().max_axis::<1>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().max_axis::<-1>();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().max_axis::<0>();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().max_axis::<1>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().max_axis::<2>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().max_axis::<-1>();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().max_axis::<0>();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().max_axis::<1>();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().max_axis::<2>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().max_axis::<3>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().max_axis::<-1>();
    }

    #[test]
    fn test_max_axis_0_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().max_axis::<0>();
        assert_eq!(r.data(), &[3.0, 2.0, 2.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [0.00000000, 2.463019, 2.463019],
                [6.695179, 0.00000000, 2.463019]
            ]
        );
    }

    #[test]
    fn test_max_axis_1_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().max_axis::<1>();
        assert_eq!(r.data(), &[2.0, 3.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 3.694528, 3.694528], [10.0427685, 0.0, 0.0]]
        );
    }
}
