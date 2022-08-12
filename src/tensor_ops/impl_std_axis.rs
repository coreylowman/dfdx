use crate::prelude::*;

/// `t.std(-1)`.Reduces the last dimension of the tensor by computing std deviation of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `(t + epsilon).var_last_dim().sqrt()`
///
/// See [var_last_dim()] and [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = std_last_dim(t, 0.0);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub fn std_axis<T, const I: isize>(t: T, epsilon: T::Dtype) -> T::Reduced
where
    T: Tensor<Dtype = f32> + Reduce1<I>,
    T::Array: HasAxis<I>,
    T::Device: ReduceAxis<T::Array, I, Reduced = <T::Reduced as HasArrayType>::Array>,
{
    sqrt(add_scalar(var_axis::<T, I>(t), epsilon))
}

/// `t.var(-1)`. Reduces the last dimension of the tensor by computing variance of all values in the last dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// Computes: `(t - t.mean_last_dim()).square().sum_last_dim() / NUM_ELEMENTS`
///
/// See [std_last_dim()] and [mean_last_dim()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = var_last_dim(t);
/// assert_eq!(r.data(), &[0.6666667, 6.0]);
/// ```
///
/// Note: equivalent to pytorch: `t.var(-1, unbiased=False)`.
pub fn var_axis<T, const I: isize>(t: T) -> T::Reduced
where
    T: Tensor<Dtype = f32> + Reduce1<I>,
    T::Array: HasAxis<I>,
    T::Device: ReduceAxis<T::Array, I, Reduced = <T::Reduced as HasArrayType>::Array>,
{
    let num_elements: f32 = <T::Array as HasAxis<I>>::SIZE as f32;
    let (t, tape) = t.split_tape();
    let mean = mean_axis::<T, I>(t.duplicate().put_tape(tape));
    div_scalar(
        sum_axis::<T, I>(square(sub(mean.broadcast_to(), &t))),
        num_elements,
    )
}

macro_rules! std_last_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [std_axis()] on `self`.
    pub fn std_axis<const I: isize>(self, epsilon: f32) -> <Self as Reduce1<I>>::Reduced
    where
        Self: Reduce1<I>,
        <Self as HasArrayType>::Array: HasAxis<I>,
        <Self as HasDevice>::Device: ReduceAxis<
            <Self as HasArrayType>::Array,
            I,
            Reduced = <<Self as Reduce1<I>>::Reduced as HasArrayType>::Array,
        >,
    {
        std_axis::<Self, I>(self, epsilon)
    }

    /// Calls [var_last_dim()] on `self`.
    pub fn var_axis<const I: isize>(self) -> <Self as Reduce1<I>>::Reduced
    where
        Self: Reduce1<I>,
        <Self as HasArrayType>::Array: HasAxis<I>,
        <Self as HasDevice>::Device: ReduceAxis<
            <Self as HasArrayType>::Array,
            I,
            Reduced = <<Self as Reduce1<I>>::Reduced as HasArrayType>::Array,
        >,
    {
        var_axis::<Self, I>(self)
    }
}
    };
}

std_last_impl!(Tensor1D, [M]);
std_last_impl!(Tensor2D, [M, N]);
std_last_impl!(Tensor3D, [M, N, O]);
std_last_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valids_sum_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().var_axis::<0>();
        let _: Tensor0D = Tensor1D::<5>::zeros().var_axis::<-1>();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().var_axis::<0>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().var_axis::<1>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().var_axis::<-1>();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().var_axis::<0>();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().var_axis::<1>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().var_axis::<2>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().var_axis::<-1>();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<0>();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<1>();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<2>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<3>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<-1>();
    }

    #[test]
    fn test_std_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 4.0, 8.0]);
        let r: Tensor0D<OwnedTape> = t.trace().std_axis::<-1>(0.0);
        assert_eq!(r.data(), &2.867442);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().sum().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[-6.8167453, -0.6816746, 7.4984202]
        );
    }

    #[test]
    fn test_std_last_2d() {
        let t: Tensor2D<2, 4> = Tensor2D::new([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().std_axis::<-1>(0.0);
        assert_eq!(r.data(), &[1.118034, 3.7666297]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [-0.16770509, -0.0559017, 0.0559017, 0.16770509],
                [-0.14104122, -0.07466887, 0.024889633, 0.19082046]
            ]
        );
    }

    #[test]
    fn test_std_last_3d() {
        let t: Tensor3D<4, 2, 2> = Tensor3D::new([
            [[1.0, 2.0], [5.0, 6.0]],
            [[-2.0, -3.0], [-4.0, -6.0]],
            [[2.0, -1.0], [-6.0, 5.0]],
            [[-2.0, 3.0], [4.0, -5.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().std_axis::<-1>(0.0);
        assert_eq!(r.data(), &[[0.5, 0.5], [0.5, 1.0], [1.5, 5.5], [2.5, 4.5]]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[-0.0625, 0.0625], [-0.0625, 0.0625]],
                [[0.0625, -0.0625], [0.0625, -0.0625]],
                [[0.0625, -0.0625], [-0.0625, 0.0625]],
                [[-0.0625, 0.0625], [0.0625, -0.0625]]
            ]
        );
    }
}
