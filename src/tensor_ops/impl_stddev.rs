use crate::arrays::{HasArrayType, HasAxes};
use crate::gradients::Tape;
use crate::prelude::*;

/// Reduces `Axes` of `T` by computing std deviation of all values in those axes.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.std(Axes, unbiased=False)`
///
/// **Related functions**: [var()], [sqrt()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.stddev(0.0);
/// assert_eq!(r.data(), &[0.6666667_f32.sqrt(), 6.0_f32.sqrt()]);
/// ```
pub fn stddev<T: Reduce<Axes>, Axes>(t: T, epsilon: T::Dtype) -> T::Reduced
where
    T::Array: HasAxes<Axes>,
{
    sqrt(add_scalar(var(t), epsilon))
}

/// Reduces `Axes` of `T` by computing variance of all values in those axes.
/// Result [Tensor] has smaller number of dimensions.
///
/// **Pytorch equivalent**: `t.var(Axes, unbiased=False)`
///
/// **Related functions**: [stddev()], [mean()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[2.0, 3.0, 4.0], [3.0, 6.0, 9.0]]);
/// let r: Tensor1D<2> = t.var();
/// assert_eq!(r.data(), &[0.6666667, 6.0]);
/// ```
pub fn var<T: Reduce<Axes>, Axes>(t: T) -> T::Reduced
where
    T::Array: HasAxes<Axes>,
{
    let num_elements: f32 = <T::Array as HasAxes<Axes>>::SIZE as f32;
    let (t, tape) = t.split_tape();
    let mean = mean(t.clone().put_tape(tape)).broadcast();
    div_scalar(sum(square(sub(mean, t))), num_elements)
}

macro_rules! impl_std_and_var {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [stddev()]
    pub fn stddev<T, Axes>(self, epsilon: f32) -> T
    where
        Self: ReduceTo<T, Axes>,
        <Self as HasArrayType>::Array: HasAxes<Axes>,
    {
        stddev(self, epsilon)
    }
    /// Calls [var()]
    pub fn var<T, Axes>(self) -> T
    where
        Self: ReduceTo<T, Axes>,
        <Self as HasArrayType>::Array: HasAxes<Axes>,
    {
        var(self)
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
        let _: Tensor0D = Tensor1D::<5>::zeros().var();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().var();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().var();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().var();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().var();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().var();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().var();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().var();
    }

    #[test]
    fn test_var_axis_0_2d() {
        let t: Tensor2D<2, 4> = tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 10.0]]);
        let r = t.trace().var::<_, Axis<0>>();
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
        let r: Tensor1D<2, OwnedTape> = t.trace().var::<_, Axis<1>>();
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
        let r = t.trace().stddev::<_, Axis<0>>(1e-8);
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
        let r: Tensor1D<2, OwnedTape> = t.trace().stddev(0.0);
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
    fn test_std_axes_2d_to_1d() {
        let t: Tensor2D<2, 3> = TensorCreator::zeros();
        let _: Tensor0D<_> = t.stddev(1e-3);
    }
}
