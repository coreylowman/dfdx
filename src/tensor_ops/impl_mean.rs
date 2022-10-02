use crate::prelude::*;

/// Average the values along `Axes` of `T`.
///
/// **Pytorch equivalent**: `t.mean(Axes)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor0D = t.mean();
/// assert_eq!(r.data(), &3.5);
/// ```
///
/// Reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor1D<2> = t.mean();
/// assert_eq!(r.data(), &[2.0, 5.0]);
/// ```
pub fn mean<T: Reduce<Axes>, Axes>(t: T) -> T::Reduced
where
    T::Array: HasAxes<Axes>,
{
    div_scalar(sum(t), <T::Array as HasAxes<Axes>>::SIZE as f32)
}

macro_rules! mean_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [mean()] with `AllAxes`
    pub fn mean<T, Axes>(self) -> T
    where
        Self: ReduceTo<T, Axes>,
        <Self as HasArrayType>::Array: HasAxes<Axes>
    {
        mean(self)
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
    use crate::tests::assert_close;
    use rand::thread_rng;

    #[test]
    fn test_valids_mean_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().mean();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().mean();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().mean();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().mean();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().mean();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().mean();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().mean();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().mean();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().mean();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().mean();
    }

    #[test]
    fn test_mean_1d() {
        let t: Tensor1D<3> = tensor([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().mean();
        assert_eq!(r.data(), &2.0);
        // NOTE: .exp() so we cover the case where .mean() has to use result grad.
        let gradients = r.exp().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[2.4630187, 2.4630187, 2.4630187]
        );
    }

    #[test]
    fn test_mean_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor0D<OwnedTape> = t.trace().mean();
        assert_eq!(r.data(), &3.5);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mean_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r: Tensor0D<OwnedTape> = t.trace().mean();
        assert_eq!(r.data(), &1.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 24.0; 3]; 2]; 4]);
    }

    #[test]
    fn test_mean_axis_0_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean::<_, Axis<0>>();
        assert_eq!(r.data(), &[-0.5, 3.0, -1.5]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.10108845, 3.3475895, 0.037188362]; 2]
        );
    }

    #[test]
    fn test_mean_axis_1_2d() {
        let t: Tensor2D<2, 3> = tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r = t.trace().mean::<_, Axis<1>>();
        assert_eq!(r.data(), &[2.0, -4.0 / 3.0]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[1.2315093; 3], [0.043932855; 3]]
        );
    }

    #[test]
    fn test_mean_axes_3d_to_1d_02() {
        let mut rng = thread_rng();
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<3, _> = t.trace().mean::<_, Axes2<0, 2>>();
        let r2 = t.trace().sum::<_, Axis<0>>().sum::<_, Axis<1>>() / 8.0;
        assert_close(r.data(), r2.data());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(g.ref_gradient(&t), &[[[1. / 24.; 4]; 3]; 2]);
        assert_close(g.ref_gradient(&t), g2.ref_gradient(&t));
    }

    #[test]
    fn test_mean_axes_3d_to_1d_01() {
        let mut rng = thread_rng();
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<4, _> = t.trace().mean::<_, Axes2<0, 1>>();
        let r2 = t.sum::<_, Axis<0>>().sum::<_, Axis<0>>() / 6.0;
        assert_close(r.data(), r2.data());
    }

    #[test]
    fn test_mean_axes_3d_to_1d_12() {
        let mut rng = thread_rng();
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<2, _> = t.trace().mean::<_, Axes2<1, 2>>();
        let r2 = t.sum::<_, Axis<1>>().sum::<_, Axis<1>>() / 12.0;
        assert_close(r.data(), r2.data());
    }

    #[test]
    fn test_mean_axes_4d_to_1d() {
        let mut rng = thread_rng();
        let t: Tensor4D<2, 3, 4, 5> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<3, _> = t.trace().mean::<_, Axes3<0, 2, 3>>();
        let r2 = t
            .sum::<_, Axis<0>>()
            .sum::<_, Axis<1>>()
            .sum::<_, Axis<1>>()
            / 40.0;
        assert_close(r.data(), r2.data());
    }
}
