use crate::prelude::*;

/// `t.mean(-1)`. Reduces the last dimension of the tensor by taking mean of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor1D<2> = t.mean_axis::<1>();
/// assert_eq!(r.data(), &[2.0, 5.0]);
/// ```
pub fn mean_axis<T: Tensor<Dtype = f32>, const I: isize>(t: T) -> T::Reduced
where
    T: Reduce1<I>,
    T::Array: HasAxis<I>,
    T::Device: ReduceAxis<T::Array, I, Reduced = <T::Reduced as HasArrayType>::Array>,
{
    div_scalar(sum_axis::<T, I>(t), <T::Array as HasAxis<I>>::SIZE as f32)
}

macro_rules! mean_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [mean_axis()] on `self`.
    pub fn mean_axis<const I: isize>(self) -> <Self as Reduce1<I>>::Reduced
    where
        Self: Reduce1<I>,
        <Self as HasArrayType>::Array: HasAxis<I>,
        <Self as HasDevice>::Device: ReduceAxis<
            <Self as HasArrayType>::Array,
            I,
            Reduced = <<Self as Reduce1<I>>::Reduced as HasArrayType>::Array,
        >,
    {
        mean_axis::<Self, I>(self)
    }
}
    };
}

mean_axis_impl!(Tensor0D, []);
mean_axis_impl!(Tensor1D, [M]);
mean_axis_impl!(Tensor2D, [M, N]);
mean_axis_impl!(Tensor3D, [M, N, O]);
mean_axis_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_last_0d() {
        let t = Tensor0D::new(2.0);
        let r: Tensor0D<OwnedTape> = t.trace().mean_axis::<-1>();
        assert_eq!(r.data(), &2.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_mean_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().mean_axis::<-1>();
        assert_eq!(r.data(), &2.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[2.4630187; 3]);
    }

    #[test]
    fn test_mean_last_2d() {
        let t: Tensor2D<2, 4> = Tensor2D::new([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().mean_axis::<-1>();
        assert_eq!(r.data(), &[2.5, 5.5]);
        let gradients = r.sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
        );
    }

    #[test]
    fn test_mean_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().mean_axis::<-1>();
        assert_eq!(
            r.data(),
            &[
                [2.0, 5.0],
                [-2.0, -5.0],
                [-2.0 / 3.0, -5.0 / 3.0],
                [2.0 / 3.0, 5.0 / 3.0]
            ]
        );
        let gradients = r.sum().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 3.0; 3]; 2]; 4]);
    }
}
