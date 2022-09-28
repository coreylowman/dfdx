use crate::prelude::*;

/// Reduces dimension `I` of `T` by computing std deviation of all values in that dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.std(I, unbiased=False)`
///
/// **Related functions**: [var_axes()], [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.std_axis::<-1>(0.0);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub fn std_axes<T, Axes>(t: T, epsilon: T::Dtype) -> T::Reduced
where
    T: Reduce<Axes>,
    T::Array: HasAxes<Axes>,
{
    sqrt(add_scalar(var_axes(t), epsilon))
}

/// Reduces dimension `I` of the tensor by computing variance of all values in the dimension.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.var(I, unbiased=False)`
///
/// **Related functions**: [std_axes()], [mean_axes()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.var_axis::<-1>();
/// assert_eq!(r.data(), &[0.6666667, 6.0]);
/// ```
pub fn var_axes<T, Axes>(t: T) -> T::Reduced
where
    T: Reduce<Axes>,
    T::Array: HasAxes<Axes>,
{
    let num_elements: f32 = <T::Array as HasAxes<Axes>>::SIZE as f32;
    let (t, tape) = t.split_tape();
    let mean = mean_axes(t.duplicate().put_tape(tape)).broadcast();
    div_scalar(sum_axes(square(sub(mean, &t))), num_elements)
}

macro_rules! impl_std_and_var {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [std_axes()] on `self` with `Axis<I>`
    pub fn std_axis<const I: isize>(self, epsilon: f32) -> <Self as Reduce<Axis<I>>>::Reduced
    where
        Self: Reduce<Axis<I>>,
        <Self as HasArrayType>::Array: HasAxes<Axis<I>>,
    {
        std_axes(self, epsilon)
    }
    /// Calls [var_axes()] on `self` with `Axis<I>`
    pub fn var_axis<const I: isize>(self) -> <Self as Reduce<Axis<I>>>::Reduced
    where
        Self: Reduce<Axis<I>>,
        <Self as HasArrayType>::Array: HasAxes<Axis<I>>,
    {
        var_axes(self)
    }
}
    };
}

impl_std_and_var!(Tensor1D, [M]);
impl_std_and_var!(Tensor2D, [M, N]);
impl_std_and_var!(Tensor3D, [M, N, O]);
impl_std_and_var!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valids_var_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().var_axis::<-1>();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().var_axis::<0>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().var_axis::<-1>();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().var_axis::<0>();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().var_axis::<1>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().var_axis::<-1>();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<0>();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<1>();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<2>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().var_axis::<-1>();
    }

    #[test]
    fn test_var_axis_0_2d() {
        let t: Tensor2D<2, 4> = tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.trace().var_axis::<0>();
        assert_eq!(r.data(), &[0.25, 0.0, 1.0, 9.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.125, 0.0, -0.25, -0.75], [-0.125, 0.0, 0.25, 0.75]]
        );
    }

    #[test]
    fn test_var_axis_1_2d() {
        let t: Tensor2D<2, 4> = tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().var_axis::<-1>();
        assert_eq!(r.data(), &[1.25, 14.1875]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [-0.375, -0.125, 0.125, 0.375],
                [-1.0625, -0.5625, 0.1875, 1.4375]
            ]
        );
    }

    #[test]
    fn test_std_axis_0_2d() {
        let t: Tensor2D<2, 4> = tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.trace().std_axis::<0>(1e-8);
        assert_eq!(r.data(), &[0.5, 0.0001, 1.0, 3.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.125, 0.0, -0.125, -0.125], [-0.125, 0.0, 0.125, 0.125]]
        );
    }

    #[test]
    fn test_std_axis_1_2d() {
        let t: Tensor2D<2, 4> = tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
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
    fn test_std_axes_3d_to_1d() {
        let t: Tensor2D<2, 3> = TensorCreator::zeros();
        let _: Tensor0D<_> = std_axes::<_, Axes2<0, 1>>(t, 1e-3);
    }
}
