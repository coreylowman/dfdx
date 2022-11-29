#![allow(clippy::type_complexity)]

mod cpu_kernel;

use crate::{arrays::*, gradients::Tape, tensor::storage::*, tensor::*};

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

/// TODO docstring
pub trait SelectAlong<T, Idx>: HasErr {
    fn select_along<Ax>(self, idx: Idx) -> T
    where
        Self: SelectTo<T, Ax, Idx>,
    {
        self.try_select_along::<Ax>(idx).unwrap()
    }
    fn try_select_along<Ax>(self, idx: Idx) -> Result<T, Self::Err>
    where
        Self: SelectTo<T, Ax, Idx>,
    {
        self.try_select(idx)
    }
}

impl<Src: HasErr, Dst, Idx> SelectAlong<Dst, Idx> for Src {}

pub trait SelectAxisKernel<E: Dtype>: DeviceStorage {
    fn forward<const I: isize, S: Shape + ReduceShape<Axis<I>>>(
        &self,
        inp: &Self::Storage<S, E>,
        idx: &Self::Storage<(), usize>,
    ) -> Result<Self::Storage<S::Reduced, E>, Self::Err>;
    fn backward<const I: isize, S: Shape + ReduceShape<Axis<I>>>(
        &self,
        grad_inp: &mut Self::Storage<S, E>,
        idx: &Self::Storage<(), usize>,
        grad_out: &Self::Storage<S::Reduced, E>,
    ) -> Result<(), Self::Err>;
}

impl<
        const I: isize,
        S: Shape + ReduceShape<Axis<I>>,
        E: Dtype,
        D: DeviceStorage + SelectAxisKernel<E>,
        T: Tape<D>,
    > SelectTo<Tensor<S::Reduced, E, D, T>, Axis<I>, Tensor<(), usize, D>> for Tensor<S, E, D, T>
{
    fn try_select(
        self,
        idx: Tensor<(), usize, D>,
    ) -> Result<Tensor<S::Reduced, E, D, T>, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(&inp.storage, &idx.storage)?;
        let out = make_tensor(&inp.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device.backward(grad_inp, &idx.storage, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

pub trait ReplaceAxisKernel<E: Dtype>: DeviceStorage {
    fn forward<const I: isize, D: Dim, S: Shape + ReplaceDim<I, D>>(
        &self,
        inp: &Self::Storage<S, E>,
        idx: &Self::Storage<(D,), usize>,
    ) -> Result<Self::Storage<S::Replaced, E>, Self::Err>;
    fn backward<const I: isize, D: Dim, S: Shape + ReplaceDim<I, D>>(
        &self,
        grad_inp: &mut Self::Storage<S, E>,
        idx: &Self::Storage<(D,), usize>,
        grad_out: &Self::Storage<S::Replaced, E>,
    ) -> Result<(), Self::Err>;
}

impl<
        const I: isize,
        New: Dim,
        S: Shape + ReplaceDim<I, New>,
        E: Dtype,
        D: DeviceStorage + ReplaceAxisKernel<E>,
        T: Tape<D>,
    > SelectTo<Tensor<S::Replaced, E, D, T>, Axis<I>, Tensor<(New,), usize, D>>
    for Tensor<S, E, D, T>
{
    fn try_select(
        self,
        idx: Tensor<(New,), usize, D>,
    ) -> Result<Tensor<S::Replaced, E, D, T>, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(&inp.storage, &idx.storage)?;
        let out = make_tensor(&inp.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device.backward(grad_inp, &idx.storage, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

pub trait SelectBatchKernel<E: Dtype>: DeviceStorage {
    fn forward<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim>(
        &self,
        inp: &Self::Storage<(S1, S2), E>,
        idx: &Self::Storage<(Batch, Seq), usize>,
    ) -> Result<Self::Storage<(Batch, Seq, S2), E>, Self::Err>;
    fn backward<Batch: Dim, Seq: Dim, S1: Dim, S2: Dim>(
        &self,
        grad_inp: &mut Self::Storage<(S1, S2), E>,
        idx: &Self::Storage<(Batch, Seq), usize>,
        grad_out: &Self::Storage<(Batch, Seq, S2), E>,
    ) -> Result<(), Self::Err>;
}

impl<
        Batch: Dim,
        Seq: Dim,
        S1: Dim,
        S2: Dim,
        E: Dtype,
        D: DeviceStorage + SelectBatchKernel<E>,
        T: Tape<D>,
    > SelectTo<Tensor<(Batch, Seq, S2), E, D, T>, Axis<0>, Tensor<(Batch, Seq), usize, D>>
    for Tensor<(S1, S2), E, D, T>
{
    fn try_select(
        self,
        idx: Tensor<(Batch, Seq), usize, D>,
    ) -> Result<Tensor<(Batch, Seq, S2), E, D, T>, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(&inp.storage, &idx.storage)?;
        let out = make_tensor(&inp.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device.backward(grad_inp, &idx.storage, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

impl<Src: Shape, Dst, Ax, E: Dtype, D: DeviceStorage, T: Tape<D>> SelectTo<Dst, Ax, usize>
    for Tensor<Src, E, D, T>
where
    Self: SelectTo<Dst, Ax, Tensor<(), usize, D>>,
    D: TensorFromArray<usize, Rank0, usize>,
{
    fn try_select(self, idx: usize) -> Result<Dst, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

impl<Src: Shape, Dst, Ax, E: Dtype, D: DeviceStorage, T: Tape<D>, const Z: usize>
    SelectTo<Dst, Ax, [usize; Z]> for Tensor<Src, E, D, T>
where
    Self: SelectTo<Dst, Ax, Tensor<Rank1<Z>, usize, D>>,
    D: TensorFromArray<[usize; Z], Rank1<Z>, usize>,
{
    fn try_select(self, idx: [usize; Z]) -> Result<Dst, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

impl<Src: Shape, Dst, Ax, E: Dtype, D: DeviceStorage, T: Tape<D>> SelectTo<Dst, Ax, &[usize]>
    for Tensor<Src, E, D, T>
where
    Self: SelectTo<Dst, Ax, Tensor<(Dyn,), usize, D>>,
    D: for<'a> TensorFromArray<&'a [usize], (Dyn,), usize>,
{
    fn try_select(self, idx: &[usize]) -> Result<Dst, Self::Err> {
        let idx = self.device.tensor(idx);
        self.try_select(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::Axis;
    use crate::tensor::storage::{AsArray, Randn};
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_select_1d_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let t_array = t.as_array();
        let r = t.trace().select(0);
        assert_eq!(r.as_array(), t_array[0]);
        let g = r.exp().backward();
        assert_eq!(g.get(&t).as_array(), [t_array[0].exp(), 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_select_1d_less_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let t_array = t.as_array();
        let r = t.trace().select([0, 3]);
        assert_eq!(r.as_array(), [t_array[0], t_array[3]]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).as_array(), [0.5, 0.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn test_select_1d_more_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let _t = t.as_array();
        let r = t.trace().select([0, 1, 2, 3, 4, 2, 4, 4]);
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
        let r = t.trace().select(2);
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
        let r = t.trace().select(dev.tensor([[2, 0, 3], [0, 0, 3]]));
        let r_array = r.as_array();
        assert_eq!(r_array[0], [t_array[2], t_array[0], t_array[3]]);
        assert_eq!(r_array[1], [t_array[0], t_array[0], t_array[3]]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).as_array(), [[3.; 5], [0.; 5], [1.; 5], [2.; 5]]);
    }
}
