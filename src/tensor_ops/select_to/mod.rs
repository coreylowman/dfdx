#![allow(clippy::type_complexity)]

mod cpu_kernel;

use crate::{gradients::Tape, shapes::*, tensor::storage::*, tensor::*};

pub trait ReplaceDimKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Src::Index, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Ax>;
}

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
pub trait SelectTo<D: DeviceStorage>: HasErr + HasShape {
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
    fn select<Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        self,
        idx: Tensor<<Self::Shape as ReplaceDimTo<Dst, Ax>>::Index, usize, D>,
    ) -> Self::WithShape<Dst>
    where
        Self::Shape: ReplaceDimTo<Dst, Ax>,
    {
        self.try_select(idx).unwrap()
    }

    fn try_select<Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        self,
        idx: Tensor<<Self::Shape as ReplaceDimTo<Dst, Ax>>::Index, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReplaceDimTo<Dst, Ax>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage + ReplaceDimKernel<E>, T: Tape<D>> SelectTo<D>
    for Tensor<Src, E, D, T>
{
    fn try_select<Dst: Shape, Ax: Axes<Array = [isize; 1]>>(
        self,
        idx: Tensor<<Self::Shape as ReplaceDimTo<Dst, Ax>>::Index, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReplaceDimTo<Dst, Ax>,
    {
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(&inp.storage, &idx.storage)?;
        let out = inp.device.upgrade(storage);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(grad_inp, &idx.storage, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_valid_selects() {
        let dev = build_test_device!();

        let t = dev.randn::<Rank1<5>>();
        let _ = t.clone().select::<Rank0, _>(dev.tensor(0));
        let _ = t.select::<Rank1<7>, _>(dev.tensor([0; 7]));

        let t = dev.randn::<Rank2<2, 3>>();
        let _ = t.clone().select::<Rank1<3>, _>(dev.tensor(0));
        let _ = t.clone().select::<Rank1<2>, _>(dev.tensor([0; 2]));
        let _ = t.clone().select::<Rank2<5, 3>, _>(dev.tensor([0; 5]));
        let _ = t.select::<Rank2<2, 5>, _>(dev.tensor([[0; 5]; 2]));
    }

    #[test]
    fn test_remove_1d_backward() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank1<5>>();
        let r = t.trace().select::<Rank0, _>(dev.tensor(0));
        let t_array = t.array();
        assert_eq!(r.array(), t_array[0]);
        let g = r.exp().backward();
        assert_eq!(g.get(&t).array(), [t_array[0].exp(), 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_replace_1d_backward() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank1<5>>();
        let r = t.trace().select::<Rank1<4>, _>(dev.tensor([0, 1, 1, 3]));
        let t_array = t.array();
        assert_eq!(r.array(), [t_array[0], t_array[1], t_array[1], t_array[3]]);
        let g = r.exp().sum().backward();
        assert_eq!(
            g.get(&t).array(),
            [
                t_array[0].exp(),
                2.0 * (t_array[1]).exp(),
                0.0,
                t_array[3].exp(),
                0.0
            ]
        );
    }

    #[test]
    fn test_replace_1d_less_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let t_array = t.array();
        let r = t.trace().select::<Rank1<2>, _>(dev.tensor([0, 3]));
        assert_eq!(r.array(), [t_array[0], t_array[3]]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [0.5, 0.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn test_select_last_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r = t.trace().select::<Rank1<2>, _>(dev.tensor(1).broadcast());
        assert_eq!(r.array(), [2.0, -2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]);
    }

    #[test]
    fn test_replace_1d_more_backward() {
        let dev = build_test_device!();
        let t: Tensor1D<5, _> = dev.randn();
        let _t = t.array();
        let r = t
            .trace()
            .select::<Rank1<8>, _>(dev.tensor([0, 1, 2, 3, 4, 2, 4, 4]));
        assert_eq!(
            r.array(),
            [_t[0], _t[1], _t[2], _t[3], _t[4], _t[2], _t[4], _t[4]]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [1.0 / 8.0, 1.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0]
        );
    }

    #[test]
    fn test_remove_3d_axis_0_backward() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let t_array = t.array();
        let r = t.trace().select::<Rank2<3, 4>, _>(dev.tensor(0));
        assert_eq!(r.array(), t_array[0]);
        let g = r.exp().mean().backward();
        let sub_g = dev.tensor(t_array[0]).exp() / 12.0;
        assert_close(&g.get(&t).array(), &[sub_g.array(), [[0.0; 4]; 3]]);
    }

    #[test]
    fn test_remove_3d_axis_1_backward() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let t_array = t.array();
        let r = t.trace().select::<Rank2<2, 4>, _>(dev.tensor([1, 2]));
        let sub_t = [t_array[0][1], t_array[1][2]];
        assert_eq!(r.array(), sub_t);
        let g = r.exp().mean().backward();
        let sub_g = dev.tensor(sub_t).exp() / 8.0;
        let sub_g = sub_g.array();
        assert_close(
            &g.get(&t).array(),
            &[
                [[0.0; 4], sub_g[0], [0.0; 4]],
                [[0.0; 4], [0.0; 4], sub_g[1]],
            ],
        );
    }

    #[test]
    fn test_remove_3d_axis_2_backward() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let t_array = t.array();
        let r = t
            .trace()
            .select::<Rank2<2, 3>, _>(dev.tensor([[2, 3, 2], [1, 1, 0]]));
        let sub_t = [
            [t_array[0][0][2], t_array[0][1][3], t_array[0][2][2]],
            [t_array[1][0][1], t_array[1][1][1], t_array[1][2][0]],
        ];
        assert_eq!(r.array(), sub_t);
        let g = r.exp().mean().backward();
        let sub_g = dev.tensor(sub_t).exp() / 6.0;
        let sub_g = sub_g.array();
        assert_close(
            &g.get(&t).array(),
            &[
                [
                    [0.0, 0.0, sub_g[0][0], 0.0],
                    [0.0, 0.0, 0.0, sub_g[0][1]],
                    [0.0, 0.0, sub_g[0][2], 0.0],
                ],
                [
                    [0.0, sub_g[1][0], 0.0, 0.0],
                    [0.0, sub_g[1][1], 0.0, 0.0],
                    [sub_g[1][2], 0.0, 0.0, 0.0],
                ],
            ],
        );
    }

    #[test]
    fn test_select_batch_backwards() {
        let dev = build_test_device!();
        let t: Tensor2D<4, 5, _> = dev.randn::<Rank2<4, 5>>();
        let t_array = t.array();
        let r = t
            .trace()
            .select::<Rank3<2, 3, 5>, _>(dev.tensor([[2, 0, 3], [0, 0, 3]]));
        let r_array = r.array();
        assert_eq!(r_array[0], [t_array[2], t_array[0], t_array[3]]);
        assert_eq!(r_array[1], [t_array[0], t_array[0], t_array[3]]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[3.; 5], [0.; 5], [1.; 5], [2.; 5]]);
    }
}
