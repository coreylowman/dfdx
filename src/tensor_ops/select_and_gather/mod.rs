#![allow(clippy::type_complexity)]

mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

pub trait ReplaceDimKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        idx: &Tensor<Idx, usize, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>;
    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        idx: &Tensor<Idx, usize, Self>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>;
}

pub trait RemoveDimKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        idx: &Tensor<Idx, usize, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>;
    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        idx: &Tensor<Idx, usize, Self>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>;
}

/// Select a single value from a single dimension, removing that dimension
/// from the shape. Equivalent to `torch.select` from pytorch.
pub trait SelectTo<D: DeviceStorage>: HasErr + HasShape {
    /// Select values given indices.
    ///
    /// The shape of the index is the shape of the tensor up to the axis you
    /// want to select from.
    ///
    /// For example, given a tensor of shape (M, N, O), here are the required
    /// index shapes to select each axis:
    /// - Axis 0: index shape ()
    /// - Axis 1: index shape (M, )
    /// - Axis 2: index shape (M, N)
    ///
    /// Here is an example selecting from a 2d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 5>, f32, _> = dev.zeros();
    ///
    /// // select from the 0th axis
    /// let idx: Tensor<Rank0, usize, _> = dev.tensor(0);
    /// let _: Tensor<Rank1<5>, f32, _> = a.clone().select(idx);
    ///
    /// // select from the 1st axis
    /// let idx: Tensor<Rank1<3>, usize, _> = dev.tensor([0, 2, 4]);
    /// let _: Tensor<Rank1<3>, f32, _> = a.select(idx);
    ///```
    fn select<Dst: Shape, Idx: Shape>(self, idx: Tensor<Idx, usize, D>) -> Self::WithShape<Dst>
    where
        Self::Shape: RemoveDimTo<Dst, Idx>,
    {
        self.try_select(idx).unwrap()
    }

    /// Fallible select
    fn try_select<Dst: Shape, Idx: Shape>(
        self,
        idx: Tensor<Idx, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: RemoveDimTo<Dst, Idx>;
}

impl<Src: Shape, E: Dtype, D: RemoveDimKernel<E>, T: Tape<E, D>> SelectTo<D>
    for Tensor<Src, E, D, T>
{
    fn try_select<Dst: Shape, Idx: Shape>(
        self,
        idx: Tensor<Idx, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: RemoveDimTo<Dst, Idx>,
    {
        self.shape().check(idx.shape());
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(&inp, &idx)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device
                .backward(&inp, grad_inp, &idx, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

/// Select multiple values from a single axis, replacing that dimension
/// with a different one. Equivalent to `torch.gather` from pytorch.
pub trait GatherTo<D: DeviceStorage>: HasErr + HasShape {
    /// Gather values given indices.
    ///
    /// The shape of the index is the shape of the tensor up to the axis you
    /// want to select from, plus the size of the new dimension.
    ///
    /// For example, given a tensor of shape (M, N, O), here are the required
    /// index shapes to gather each axis:
    /// - Axis 0: index shape (Z, )
    /// - Axis 1: index shape (M, Z)
    /// - Axis 2: index shape (M, N, Z)
    ///
    /// where `Z` is the new dimension.
    ///
    /// Here is an example gathering from a 2d tensor:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 5>, f32, _> = dev.zeros();
    ///
    /// // gather from the 0th axis; dimension 0 becomes 4
    /// let idx: Tensor<Rank1<4>, usize, _> = dev.tensor([0, 0, 1, 2]);
    /// let _: Tensor<Rank2<4, 5>, f32, _> = a.clone().gather(idx);
    ///
    /// // gather from the 1st axis; dimension 1 becomes 2
    /// let idx: Tensor<Rank2<3, 2>, usize, _> = dev.tensor([[0, 1], [2, 3], [4, 4]]);
    /// let _: Tensor<Rank2<3, 2>, f32, _> = a.gather(idx);
    ///```
    fn gather<Dst: Shape, Idx: Shape>(self, idx: Tensor<Idx, usize, D>) -> Self::WithShape<Dst>
    where
        Self::Shape: ReplaceDimTo<Dst, Idx>,
    {
        self.try_gather(idx).unwrap()
    }

    fn try_gather<Dst: Shape, Idx: Shape>(
        self,
        idx: Tensor<Idx, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReplaceDimTo<Dst, Idx>;
}

impl<Src: Shape, E: Dtype, D: ReplaceDimKernel<E>, T: Tape<E, D>> GatherTo<D>
    for Tensor<Src, E, D, T>
{
    fn try_gather<Dst: Shape, Idx: Shape>(
        self,
        idx: Tensor<Idx, usize, D>,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReplaceDimTo<Dst, Idx>,
    {
        self.shape().check(idx.shape());
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(&inp, &idx)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device
                .backward(&inp, grad_inp, &idx, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    #[test]
    #[should_panic = "dimension 0 not the same"]
    fn test_select_wrong_index_shape_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.sample_like(&(5, 3), rand_distr::StandardNormal);
        // here we are selecting from axis 1, so the 7 should actually be a 5
        let _ = t.leaky_trace().select(dev.zeros_like(&(7,)));
    }

    #[test]
    #[should_panic = "dimension 1 not the same"]
    fn test_select_wrong_index_shape_3d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.sample_like(&(7, 5, 3), rand_distr::StandardNormal);
        let _ = t.leaky_trace().select(dev.zeros_like(&(7, 4)));
    }

    #[cfg(not(feature = "test-cuda"))]
    #[test]
    #[should_panic = "Index out of bounds: index=[7]"]
    fn test_select_index_out_of_bounds() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let _ = t.leaky_trace().select(dev.tensor(7));
    }

    #[test]
    #[should_panic = "dimension 0 not the same"]
    fn test_gather_wrong_index_shape_3d1() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.sample_like(&(5, 3, 1), rand_distr::StandardNormal);
        let r = t.leaky_trace().gather(dev.zeros_like(&(7,)));
        assert_eq!(r.shape(), &(7, 3, 1));
        let _ = t.leaky_trace().gather(dev.zeros_like(&(7, 4)));
    }

    #[test]
    #[should_panic = "dimension 1 not the same"]
    fn test_gather_wrong_index_shape_3d2() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.sample_like(&(5, 3, 1), rand_distr::StandardNormal);
        let _ = t.leaky_trace().gather(dev.zeros_like(&(5, 4, 2)));
    }

    #[cfg(not(feature = "test-cuda"))]
    #[test]
    #[should_panic = "Index out of bounds: index=[7]"]
    fn test_gather_index_out_of_bounds() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let _ = t.leaky_trace().gather(dev.tensor([7, 6, 1, 2]));
    }

    #[cfg(not(feature = "test-cuda"))]
    #[test]
    #[should_panic = "Index out of bounds: index=[5, 0]"]
    fn test_gather_batch_out_of_bounds() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<4, 5>, TestDtype, _> = dev.sample_normal();
        let _ = t
            .leaky_trace()
            .try_gather(dev.tensor([[5, 0, 0], [0, 0, 0]]));
    }

    #[test]
    fn test_select_1d_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let r = t.leaky_trace().select(dev.tensor(0));
        let t_array = t.array();
        assert_eq!(r.array(), t_array[0]);
        let g = r.exp().backward();
        assert_eq!(g.get(&t).array(), [t_array[0].exp(), 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gather_1d_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let r = t.leaky_trace().gather(dev.tensor([0, 1, 1, 3]));
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
    fn test_gather_1d_less_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r = t.leaky_trace().gather(dev.tensor([0, 3]));
        assert_eq!(r.array(), [t_array[0], t_array[3]]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [0.5, 0.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn test_gather_1d_more_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let _t = t.array();
        let r = t.leaky_trace().gather(dev.tensor([0, 1, 2, 3, 4, 2, 4, 4]));
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
    fn test_select_2d_axis_0() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r = t.leaky_trace().select(dev.tensor(0));
        assert_eq!(r.array(), [1.0, 2.0, 3.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [[1.0 / 3.0; 3], [0.0; 3]]);
    }

    #[test]
    fn test_select_2d_axis_1() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r = t.leaky_trace().select(dev.tensor([1, 1]));
        assert_eq!(r.array(), [2.0, -2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]);
    }

    #[test]
    fn test_select_2d_broadcasted() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0]);
        let r = t
            .leaky_trace()
            .broadcast::<Rank2<2, 3>, _>()
            .select(dev.tensor([0, 1]));
        assert_eq!(r.array(), [1.0, 2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_gather_2d_broadcasted() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0]);
        let idx: Tensor<Rank2<2, 2>, usize, _> = dev.tensor([[0, 1], [1, 2]]);
        let r: Tensor<Rank2<2, 2>, _, _, _> =
            t.leaky_trace().broadcast::<Rank2<2, 3>, _>().gather(idx);
        assert_eq!(r.array(), [[1.0, 2.0], [2.0, 3.0]]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [0.25, 0.5, 0.25]);
    }

    #[test]
    fn test_select_3d_axis_0_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r = t.leaky_trace().select(dev.tensor(0));
        assert_eq!(r.array(), t_array[0]);
        let g = r.exp().mean().backward();
        let sub_g = dev.tensor(t_array[0]).exp() / 12.0;
        assert_close(&g.get(&t).array(), &[sub_g.array(), [[0.0; 4]; 3]]);
    }

    #[test]
    fn test_select_3d_axis_1_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r = t.leaky_trace().select(dev.tensor([1, 2]));
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
    fn test_select_3d_axis_2_backward() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r = t.leaky_trace().select(dev.tensor([[2, 3, 2], [1, 1, 0]]));
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
    fn test_gather_batch_backwards() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<4, 5>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r = t.leaky_trace().gather(dev.tensor([[2, 0, 3], [0, 0, 3]]));
        let r_array = r.array();
        assert_eq!(r_array[0], [t_array[2], t_array[0], t_array[3]]);
        assert_eq!(r_array[1], [t_array[0], t_array[0], t_array[3]]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[3.; 5], [0.; 5], [1.; 5], [2.; 5]]);
    }

    #[test]
    fn test_gather_smaller_output_row() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let t_array = t.array();
        let r: Tensor<Rank2<2, 2>, _, _> = t.gather(dev.tensor([[0, 1], [0, 1]]));
        let r_array = r.array();
        assert_eq!(r_array[0][..], t_array[0][..2]);
        assert_eq!(r_array[1][..], t_array[1][..2]);
    }
}
