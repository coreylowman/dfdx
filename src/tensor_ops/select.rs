use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Select values along a single axis `I` resulting in `T`. Equivalent
/// to `torch.select` and `torch.gather` from pytorch.
///
/// There are two ways to select:
/// 1. Select a single value from an axis, which removes that axis and
/// returns a smaller tensor
/// 2. Select multiple values from an axis, which keeps the number
/// of dimensions the same. You can select the same element multiple
/// number of times.
pub trait Select1<T, const I: isize> {
    type Indices: Clone;

    /// Select sub elements using [Self::Indices].
    /// The same element can be selected multiple times depending
    /// on [Self::Indices].
    ///
    /// Selecting single value from 2d tensors:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// // select a single element from the 0th axis
    /// let _: Tensor1D<5> = Tensor2D::<3, 5>::zeros().select(&0);
    ///
    /// // select a single element from the 1st axis - number of elements is equal
    /// // to the size of the 0th axis, and the usize values can be 0..5
    /// let _: Tensor1D<3> = Tensor2D::<3, 5>::zeros().select(&[0, 2, 4]);
    ///```
    ///
    /// Selecting multiple values from 2d tensors:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// // select a multiple elements from the 0th axis.
    /// // the number of indices is the new size of the 0th axis.
    /// let _: Tensor2D<6, 5> = Tensor2D::<3, 5>::zeros().select(&[0, 1, 2, 0, 1, 2]);
    ///
    /// // select a multiple elements from the 1st axis.
    /// // must have same number of elements as the 0th axis, and the number of indices
    /// // is the new size of the 1st axis.
    /// let _: Tensor2D<3, 2> = Tensor2D::<3, 5>::zeros().select(&[[0, 4], [1, 3], [2, 2]]);
    /// ```
    fn select(self, indices: &Self::Indices) -> T;
}

macro_rules! impl_select {
    ($Axis:expr, $Mode:ty, $SrcTy:ty, $IndTy:tt, $DstTy:ty, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )* H: Tape> Select1<$DstTy, $Axis> for $SrcTy {
    type Indices = $IndTy;
    fn select(self, indices: &Self::Indices) -> $DstTy {
        select::<_, _, _, $Mode>(self, indices)
    }
}
    };
}

// 1d
impl_select!(-1, SelectAx0, Tensor1D<M, H>, usize, Tensor0D<H>, {M});
impl_select!(-1, SelectAx0, Tensor1D<M, H>, [usize; Z], Tensor1D<Z, H>, {M, Z});

// 2d
impl_select!(0, SelectAx0, Tensor2D<M, N, H>, usize, Tensor1D<N, H>, {M, N});
impl_select!(0, SelectAx0, Tensor2D<M, N, H>, [usize; Z], Tensor2D<Z, N, H>, {M, N, Z});
impl_select!(-1, SelectAx1, Tensor2D<M, N, H>, [usize; M], Tensor1D<M, H>, {M, N});
impl_select!(-1, SelectAx1, Tensor2D<M, N, H>, [[usize; Z]; M], Tensor2D<M, Z, H>, {M, N, Z});

// 3d
impl_select!(0, SelectAx0, Tensor3D<M, N, O, H>, usize, Tensor2D<N, O, H>, {M, N, O});
impl_select!(0, SelectAx0, Tensor3D<M, N, O, H>, [usize; Z], Tensor3D<Z, N, O, H>, {M, N, O, Z});
impl_select!(1, SelectAx1, Tensor3D<M, N, O, H>, [usize; M], Tensor2D<M, O, H>, {M, N, O});
impl_select!(1, SelectAx1, Tensor3D<M, N, O, H>, [[usize; Z]; M], Tensor3D<M, Z, O, H>, {M, N, O, Z});
impl_select!(-1, SelectAx2, Tensor3D<M, N, O, H>, [[usize; N]; M], Tensor2D<M, N, H>, {M, N, O});
impl_select!(-1, SelectAx2, Tensor3D<M, N, O, H>, [[[usize; Z]; N]; M], Tensor3D<M, N, Z, H>, {M, N, O, Z});

// 4d
impl_select!(0, SelectAx0, Tensor4D<M, N, O, P, H>, usize, Tensor3D<N, O, P, H>, {M, N, O, P});
impl_select!(0, SelectAx0, Tensor4D<M, N, O, P, H>, [usize; Z], Tensor4D<Z, N, O, P, H>, {M, N, O, P, Z});
impl_select!(1, SelectAx1, Tensor4D<M, N, O, P, H>, [usize; M], Tensor3D<M, O, P, H>, {M, N, O, P});
impl_select!(1, SelectAx1, Tensor4D<M, N, O, P, H>, [[usize; Z]; M], Tensor4D<M, Z, O, P, H>, {M, N, O, P, Z});
impl_select!(2, SelectAx2, Tensor4D<M, N, O, P, H>, [[usize; N]; M], Tensor3D<M, N, P, H>, {M, N, O, P});
impl_select!(2, SelectAx2, Tensor4D<M, N, O, P, H>, [[[usize; Z]; N]; M], Tensor4D<M, N, Z, P, H>, {M, N, O, P, Z});
impl_select!(-1, SelectAx3, Tensor4D<M, N, O, P, H>, [[[usize; O]; N]; M], Tensor3D<M, N, O, H>, {M, N, O, P});
impl_select!(-1, SelectAx3, Tensor4D<M, N, O, P, H>, [[[[usize; Z]; O]; N]; M], Tensor4D<M, N, O, Z, H>, {M, N, O, P, Z});

/// Select batched values from axis 0, resulting in `T`. Equivalent
/// to `torch.select` and `torch.gather` from pytorch.
pub trait SelectBatchAx0<T> {
    type Indices;

    /// Select sub elements using [Self::Indices].
    /// The same element can be selected multiple times depending
    /// on [Self::Indices].
    ///
    /// This results in a tensor 1 dimension larger than self.
    ///
    /// Selecting batch of values from a 1d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor2D<2, 1> = Tensor1D::<5>::zeros().select_batch(&[[0], [1]]);
    ///```
    ///
    /// Selecting batch of values from a 2d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor3D<2, 1, 5> = Tensor2D::<3, 5>::zeros().select_batch(&[[[0], [1]], [[2], [3]]]);
    ///```
    fn select_batch(self, indices: &Self::Indices) -> T;
}

macro_rules! impl_select_batch {
    ($SrcTy:ty, $IndTy:tt, $DstTy:ty, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )* H: Tape> SelectBatchAx0<$DstTy> for $SrcTy {
    type Indices = $IndTy;
    fn select_batch(self, indices: &Self::Indices) -> $DstTy {
        select::<_, _, _, BSelectAx0>(self, indices)
    }
}
    };
}

impl_select_batch!(Tensor1D<M, H>, [[usize; Z]; B], Tensor2D<B, Z, H>, {M, B, Z});
impl_select_batch!(Tensor2D<M, N, H>, [[usize; Z]; B], Tensor3D<B, Z, N, H>, {M, N, B, Z});
impl_select_batch!(Tensor3D<M, N, O, H>, [[usize; Z]; B], Tensor4D<B, Z, N, O, H>, {M, N, O, B, Z});

pub(crate) fn select<T, I, R, Mode>(t: T, indices: &I) -> R
where
    T: Tensor<Dtype = f32>,
    I: 'static + Clone,
    R: Tensor<Dtype = f32, Tape = T::Tape>,
    <T as HasDevice>::Device: DeviceSelect<T::Array, I, Mode, Result = R::Array>,
{
    let mut result: <R as Tensor>::NoTape = TensorCreator::zeros();
    <T as HasDevice>::Device::select_axis(t.data(), indices, result.mut_data());

    #[allow(clippy::clone_on_copy)]
    let i = indices.clone();

    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        <T as HasDevice>::Device::fill(t.mut_data(), &mut |v| *v = 0.0);
        <T as HasDevice>::Device::select_add(t.mut_data(), &i, result_grad);
        <T as HasDevice>::Device::add(t_grad, t.data());
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::thread_rng;

    #[test]
    fn test_valid_selects_1d() {
        let _: Tensor0D = Tensor1D::<5>::zeros().select(&0);
        let _: Tensor1D<3> = Tensor1D::<5>::zeros().select(&[1, 2, 3]);
        let _: Tensor1D<10> = Tensor1D::<5>::zeros().select(&[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_valid_select_batches() {
        let _: Tensor2D<2, 1> = Tensor1D::<5>::zeros().select_batch(&[[0], [1]]);
        let _: Tensor3D<2, 1, 5> = Tensor2D::<3, 5>::zeros().select_batch(&[[0], [1]]);
        let _: Tensor4D<2, 1, 3, 5> = Tensor3D::<1, 3, 5>::zeros().select_batch(&[[0], [0]]);
    }

    #[test]
    fn test_select_1d_backward() {
        let mut rng = thread_rng();
        let t: Tensor1D<5> = TensorCreator::randn(&mut rng);
        let r: Tensor0D<OwnedTape> = t.trace().select(&0);
        assert_eq!(r.data(), &t.data()[0]);
        let g = r.exp().mean().backward();
        assert_eq!(g.ref_gradient(&t), &[t.data()[0].exp(), 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_select_1d_less_backward() {
        let mut rng = thread_rng();
        let t: Tensor1D<5> = TensorCreator::randn(&mut rng);
        let r: Tensor1D<2, OwnedTape> = t.trace().select(&[0, 3]);
        assert_eq!(r.data(), &[t.data()[0], t.data()[3]]);
        let g = r.mean().backward();
        assert_eq!(g.ref_gradient(&t), &[0.5, 0.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn test_select_1d_more_backward() {
        let mut rng = thread_rng();
        let t: Tensor1D<5> = TensorCreator::randn(&mut rng);
        let _t = *t.data();
        let r: Tensor1D<8, OwnedTape> = t.trace().select(&[0, 1, 2, 3, 4, 2, 4, 4]);
        assert_eq!(
            r.data(),
            &[_t[0], _t[1], _t[2], _t[3], _t[4], _t[2], _t[4], _t[4]]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.ref_gradient(&t),
            &[1.0 / 8.0, 1.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0]
        );
    }

    #[test]
    fn test_select_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().select(&2);
        assert_eq!(r.data(), &3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_select_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().select(&[1, 2]);
        assert_eq!(r.data(), &[2.0, -3.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
        );
    }

    #[test]
    fn test_select_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().select(&[[0, 1], [2, 2], [1, 1], [0, 0]]);
        assert_eq!(
            r.data(),
            &[[1.0, 5.0], [-3.0, -6.0], [2.0, 5.0], [1.0, 4.0]]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[0.125, 0.0, 0.0], [0.0, 0.125, 0.0]],
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]],
                [[0.0, 0.125, 0.0], [0.0, 0.125, 0.0]],
                [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0]]
            ]
        );
    }

    #[test]
    fn test_select_batch_backwards() {
        let mut rng = thread_rng();
        let t: Tensor2D<4, 5> = TensorCreator::randn(&mut rng);
        let r: Tensor3D<2, 3, 5, _> = t.trace().select_batch(&[[2, 0, 3], [0, 0, 3]]);
        let r0: Tensor2D<3, 5> = t.clone().select(&[2, 0, 3]);
        let r1: Tensor2D<3, 5> = t.clone().select(&[0, 0, 3]);
        assert_close(&r.data()[0], r0.data());
        assert_close(&r.data()[1], r1.data());

        let g = r.sum().backward();
        assert_eq!(g.ref_gradient(&t), &[[3.; 5], [0.; 5], [1.; 5], [2.; 5]]);
    }
}
