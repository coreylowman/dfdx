use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Reduces the last dimension of the tensor by gathering the maximum value from that dimension.
/// Resulting [Tensor] has the last dimension removed (e.g. a 2d tensor will become 1d).
///
/// **Pytorch equivalent**: `t.amax(-1)`
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
pub fn max_axis<T: Tensor<Dtype = f32>, const I: isize>(mut t: T) -> T::Reduced
where
    T: Reduce1<I>,
    T::Device: ReduceAxis<T::Array, I, Reduced = <T::Reduced as HasArrayType>::Array>,
{
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    <T::Device as ReduceAxis<T::Array, I>>::reduce_into(t.data(), result.mut_data(), f32::max);

    // store derivative in t
    T::Device::foreach_br(t.mut_data(), result.data(), &mut |l, r| {
        *l = if l == r { 1.0 } else { 0.0 }
    });

    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);

        T::Device::foreach_br(t.mut_data(), result_grad, &mut |d, r| {
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
        <Self as HasDevice>::Device: ReduceAxis<
            <Self as HasArrayType>::Array,
            I,
            Reduced = <<Self as Reduce1<I>>::Reduced as HasArrayType>::Array,
        >,
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
    fn test_max_last_0d() {
        let t = Tensor0D::new(2.0);
        let r: Tensor0D<OwnedTape> = t.trace().max_axis::<-1>();
        assert_eq!(r.data(), &2.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_max_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().max_axis::<-1>();
        assert_eq!(r.data(), &3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_max_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().max_axis::<-1>();
        assert_eq!(r.data(), &[3.0, -1.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 0.0, 0.5], [0.5, 0.0, 0.0]]
        );
    }

    #[test]
    fn test_max_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().max_axis::<-1>();
        assert_eq!(
            r.data(),
            &[[3.0, 6.0], [-1.0, -4.0], [2.0, 5.0], [3.0, 6.0]]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]],
                [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0]],
                [[0.0, 0.125, 0.0], [0.0, 0.125, 0.0]],
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]]
            ]
        );
    }
}
