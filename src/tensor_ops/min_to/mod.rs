mod cpu_kernel;

use crate::{gradients::Tape, shapes::*, tensor::storage::*, tensor::*};

use super::Device;

pub trait MinReduceKernel<E: Dtype>: DeviceStorage {
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

/// Reduces `Axes` of the tensor by gathering the minimum value from the axes.
///
/// **Pytorch equivalent**: `t.amin(Axes)`
///
/// **NOTE** This evenly distributes gradients between all equal minimum values, instead
/// of only exactly 1 value.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = t.min();
/// assert_eq!(r.data(), &[1.0, -3.0]);
/// ```
///
/// Reducing 2 axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor0D = t.min();
/// assert_eq!(r.data(), &-3.0);
/// ```
pub trait MinInto<T, Ax>: HasErr {
    fn min(self) -> T {
        self.try_min().unwrap()
    }
    fn try_min(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape, Ax: Axes>
    MinInto<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    D: MinReduceKernel<E>,
    Src: ReduceShapeTo<Dst, Ax>,
{
    fn try_min(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
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

pub trait MinTo<Ax: Axes>: HasShape + HasErr {
    fn min_to<Dst: Shape>(self) -> Self::WithShape<Dst>
    where
        Self: MinInto<Self::WithShape<Dst>, Ax>,
    {
        self.min()
    }

    fn try_min_to<Dst: Shape>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self: MinInto<Self::WithShape<Dst>, Ax>,
    {
        self.try_min()
    }
}
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Ax: Axes> MinTo<Ax> for Tensor<S, E, D, T> {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn min_along<Ax: Axes>(self) -> Tensor<S::Reduced, E, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_min_along().unwrap()
    }

    pub fn try_min_along<Ax: Axes>(self) -> Result<Tensor<S::Reduced, E, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_valids_min_axis() {
        let dev = build_test_device!();
        let _ = dev.zeros::<Rank1<5>>().min_to::<Rank0>();
        let _ = dev.zeros::<Rank2<5, 3>>().min_to::<Rank1<3>>();
        let _ = dev.zeros::<Rank2<5, 3>>().min_to::<Rank1<5>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().min_to::<Rank2<5, 3>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().min_to::<Rank2<7, 3>>();
        let _ = dev.zeros::<Rank3<7, 5, 3>>().min_to::<Rank2<7, 5>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().min_to::<Rank3<7, 5, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().min_to::<Rank3<9, 5, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().min_to::<Rank3<9, 7, 3>>();
        let _ = dev.zeros::<Rank4<9, 7, 5, 3>>().min_to::<Rank3<9, 7, 5>>();
    }

    #[test]
    fn test_min_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().min_to::<Rank1<3>>();
        assert_eq!(r.array(), [1.0, -2.0, 2.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [[0.90609396, 0.0, 2.463019], [0.0, 0.04511176, 2.463019]]
        );
    }

    #[test]
    fn test_min_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r = t.trace().min_to::<Rank1<2>>();
        assert_eq!(r.array(), [1.0, -2.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_min_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<2, 3, 4>>();
        let r = t.trace().min_to::<Rank1<4>>();
        let r2 = t.trace().min_to::<Rank2<3, 4>>().min_to::<Rank1<4>>();
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }
}
