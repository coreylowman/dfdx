mod cpu_kernel;

use crate::{
    arrays::{Dtype, Dyn, Rank0, Rank1, Shape},
    devices::{Device, HasErr, UnaryKernel},
    gradients::Tape,
    tensor::{Tensor, TensorSugar},
};

use super::utils::try_unary_op;

/// Select values along `Axes` resulting in `T`. Equivalent
/// to `torch.select` and `torch.gather` from pytorch.
///
/// There are two ways to select:
/// 1. Select a single value from an axis, which removes that axis and
/// returns a smaller tensor
/// 2. Select multiple values from an axis, which keeps the number
/// of dimensions the same. You can select the same element multiple
/// number of times.
///
/// You can also select batches of data with this trait.
pub trait SelectTo<T, Axes, Idx>: HasErr {
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
    ///
    /// Selecting batch of values from a 1d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor2D<2, 1> = Tensor1D::<5>::zeros().select(&[[0], [1]]);
    ///```
    ///
    /// Selecting batch of values from a 2d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let _: Tensor3D<2, 1, 5> = Tensor2D::<3, 5>::zeros().select(&[[0], [1]]);
    ///```
    fn select(self, idx: Idx) -> T {
        self.try_select(idx).unwrap()
    }
    fn try_select(self, idx: Idx) -> Result<T, Self::Err>;
}

pub trait SelectAlong<T, Idx>: HasErr {
    fn select_along<Axes>(self, idx: Idx) -> T
    where
        Self: SelectTo<T, Axes, Idx>,
    {
        self.select(idx)
    }
    fn try_select_along<Axes>(self, idx: Idx) -> Result<T, Self::Err>
    where
        Self: SelectTo<T, Axes, Idx>,
    {
        self.try_select(idx)
    }
}

impl<Src: HasErr, T, Idx> SelectAlong<T, Idx> for Src {}

impl<Src: Shape, Dst: Shape, Axes, E: Dtype, D: Device, T: Tape<D>>
    SelectTo<Tensor<Dst, E, D, T>, Axes, usize> for Tensor<Src, E, D, T>
where
    Self: SelectTo<Tensor<Dst, E, D, T>, Axes, Tensor<(), usize, D>>,
    D: TensorSugar<usize, Rank0, usize>,
{
    fn try_select(self, idx: usize) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

impl<Src: Shape, Dst: Shape, Axes, E: Dtype, D: Device, T: Tape<D>, const Z: usize>
    SelectTo<Tensor<Dst, E, D, T>, Axes, [usize; Z]> for Tensor<Src, E, D, T>
where
    Self: SelectTo<Tensor<Dst, E, D, T>, Axes, Tensor<Rank1<Z>, usize, D>>,
    D: TensorSugar<[usize; Z], Rank1<Z>, usize>,
{
    fn try_select(self, idx: [usize; Z]) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

impl<Src: Shape, Dst: Shape, Axes, E: Dtype, D: Device, T: Tape<D>>
    SelectTo<Tensor<Dst, E, D, T>, Axes, &[usize]> for Tensor<Src, E, D, T>
where
    Self: SelectTo<Tensor<Dst, E, D, T>, Axes, Tensor<(Dyn,), usize, D>>,
    D: for<'a> TensorSugar<&'a [usize], (Dyn,), usize>,
{
    fn try_select(self, idx: &[usize]) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SelectKernelOp<Dst, Axis, I: Shape, D: Device> {
    dst: Dst,
    indices: D::Storage<I, usize>,
    marker: std::marker::PhantomData<Axis>,
}

impl<
        Src: Shape,
        Dst: Shape + Default,
        Axes: 'static + Copy,
        Idx: Shape,
        E: Dtype,
        D: Device,
        T: Tape<D>,
    > SelectTo<Tensor<Dst, E, D, T>, Axes, Tensor<Idx, usize, D>> for Tensor<Src, E, D, T>
where
    D: UnaryKernel<SelectKernelOp<Dst, Axes, Idx, D>, Src, Dst, E>,
{
    fn try_select(self, idx: Tensor<Idx, usize, D>) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        try_unary_op(
            SelectKernelOp {
                dst: Default::default(),
                indices: idx.storage,
                marker: std::marker::PhantomData,
            },
            self,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::Axis;
    use crate::devices::{AsArray, Randn};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_select_1d_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let t_array = t.as_array();
        let r: Tensor0D<_, _> = t.trace().select(0);
        assert_eq!(r.as_array(), t_array[0]);
        let g = r.exp().backward();
        assert_eq!(g.get(&t).as_array(), [t_array[0].exp(), 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_select_1d_less_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let t_array = t.as_array();
        let r: Tensor1D<2, _, _> = t.trace().select([0, 3]);
        assert_eq!(r.as_array(), [t_array[0], t_array[3]]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).as_array(), [0.5, 0.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn test_select_1d_more_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let _t = t.as_array();
        let r: Tensor1D<8, _, _> = t.trace().select([0, 1, 2, 3, 4, 2, 4, 4]);
        assert_eq!(
            r.as_array(),
            [_t[0], _t[1], _t[2], _t[3], _t[4], _t[2], _t[4], _t[4]]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [1.0 / 8.0, 1.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0]
        );
    }

    #[test]
    fn test_select_last_1d() {
        let dev = build_test_device!();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r: Tensor0D<_, _> = t.trace().select(2);
        assert_eq!(r.as_array(), 3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let g = r.exp().backward();
        assert_eq!(g.get(&t).as_array(), [0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_select_last_2d() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, _, _> = t.trace().select_along::<Axis<1>>(1);
        assert_eq!(r.as_array(), [2.0, -2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).as_array(), [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]);
    }

    #[test]
    fn test_select_batch_backwards() {
        let dev = build_test_device!();
        let t: Tensor2D<4, 5, _> = dev.randn();
        let t_array = t.as_array();
        let r: Tensor3D<2, 3, 5, _, _> = t.trace().select(dev.tensor([[2, 0, 3], [0, 0, 3]]));
        let r_array = r.as_array();
        assert_eq!(r_array[0], [t_array[2], t_array[0], t_array[3]]);
        assert_eq!(r_array[1], [t_array[0], t_array[0], t_array[3]]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).as_array(), [[3.; 5], [0.; 5], [1.; 5], [2.; 5]]);
    }
}
