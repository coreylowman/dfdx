mod cpu_kernel;

use crate::{gradients::Tape, shapes::*, tensor::storage::*, tensor::*};

pub trait MaxReduceKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        out: &Self::Storage<Dst, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>;
}

/// Reduces `Axes` of the tensor by gathering the maximum value from that dimension.
///
/// **Pytorch equivalent**: `t.amax(Axes)`
///
/// **NOTE** This evenly distributes gradients between all equal maximum values, instead
/// of only exactly 1 value.
///
/// Example reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = t.max();
/// assert_eq!(r.data(), &[3.0, -1.0]);
/// ```
///
/// Reducing 2 axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor0D = t.max();
/// assert_eq!(r.data(), &3.0);
/// ```
pub trait MaxTo: HasErr + HasShape {
    fn max<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        self.try_max().unwrap()
    }
    fn try_max<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage + MaxReduceKernel<E>, T: Tape<D>> MaxTo
    for Tensor<S, E, D, T>
{
    fn try_max<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ReduceShapeTo<Dst, Ax>,
    {
        let dst: Dst = self.shape().reduced();
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward(dst, &inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device
                .backward(&inp.storage, grad_inp, &phantom_out.storage, grad_out)
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
    fn test_valids_max_axis() {
        let dev = build_test_device!();
        let _ = dev.zeros::<Rank1<5>>().max::<Rank0, _>();
        let _ = dev.zeros::<Rank2<5, 3>>().max::<Rank1<3>, _>();
        let _ = dev.zeros::<Rank2<5, 3>>().max::<Rank1<5>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max::<Rank2<5, 3>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max::<Rank2<7, 3>, _>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max::<Rank2<7, 5>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max::<Rank3<7, 5, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max::<Rank3<9, 5, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max::<Rank3<9, 7, 3>, _>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max::<Rank3<9, 7, 5>, _>();
    }

    #[test]
    fn test_max_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().max::<_, Axis<0>>();
        assert_eq!(r.array(), [3.0, 2.0, 2.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [[0.0, 2.463019, 2.463019], [6.695179, 0.0, 2.463019]]
        );
    }

    #[test]
    fn test_max_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().max::<_, Axis<1>>();
        assert_eq!(r.array(), [2.0, 3.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_max_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().max::<Rank1<4>, _>();
        let r2 = t.trace().max::<_, Axis<0>>().max::<_, Axis<0>>();
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }
}
