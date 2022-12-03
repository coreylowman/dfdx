mod cpu_kernel;

use crate::{arrays::*, gradients::Tape, tensor::storage::*, tensor::*};

use super::Device;

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
pub trait MaxInto<T, Ax>: HasErr {
    fn max(self) -> T {
        self.try_max().unwrap()
    }
    fn try_max(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape, Ax: Axes>
    MaxInto<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    D: MaxReduceKernel<E>,
    Src: ReduceShapeTo<Dst, Ax>,
{
    fn try_max(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let dst: Dst = self.shape().reduced();
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(dst, &inp.storage)?;
        let out = inp.device.upgrade(storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device
                .backward(&inp.storage, grad_inp, &phantom_out.storage, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

pub trait MaxTo<Ax: Axes>: HasShape + HasErr {
    fn max_to<Dst: Shape>(self) -> Self::WithShape<Dst>
    where
        Self: MaxInto<Self::WithShape<Dst>, Ax>,
    {
        self.max()
    }

    fn try_max_to<Dst: Shape>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self: MaxInto<Self::WithShape<Dst>, Ax>,
    {
        self.try_max()
    }
}
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Ax: Axes> MaxTo<Ax> for Tensor<S, E, D, T> {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn max_along<Ax: Axes>(self) -> Tensor<S::Reduced, E, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_max_along().unwrap()
    }

    pub fn try_max_along<Ax: Axes>(self) -> Result<Tensor<S::Reduced, E, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_max()
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
        let _ = dev.zeros::<Rank1<5>>().max_to::<Rank0>();
        let _ = dev.zeros::<Rank2<5, 3>>().max_to::<Rank1<3>>();
        let _ = dev.zeros::<Rank2<5, 3>>().max_to::<Rank1<5>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max_to::<Rank2<5, 3>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max_to::<Rank2<7, 3>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().max_to::<Rank2<7, 5>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max_to::<Rank3<7, 5, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max_to::<Rank3<9, 5, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max_to::<Rank3<9, 7, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().max_to::<Rank3<9, 7, 5>>();
    }

    #[test]
    fn test_max_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().max_to::<Rank1<3>>();
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
        let r = t.trace().max_to::<Rank1<2>>();
        assert_eq!(r.array(), [2.0, 3.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_max_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().max_to::<Rank1<4>>();
        let r2 = t.trace().max_to::<Rank2<3, 4>>().max_to::<Rank1<4>>();
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }
}
