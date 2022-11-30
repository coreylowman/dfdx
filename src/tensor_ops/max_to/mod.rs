mod cpu_kernel;

use crate::{arrays::*, gradients::Tape, tensor::storage::*, tensor::*};

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
pub trait MaxTo<T, Ax>: HasErr {
    fn max(self) -> T {
        self.try_max().unwrap()
    }
    fn try_max(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape + Default, Ax: Axes>
    MaxTo<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    D: MaxReduceKernel<E>,
    Src: ReduceShapeTo<Dst, Ax>,
{
    fn try_max(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let dst: Dst = Default::default();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_valids_max_axis() {
        let dev = build_test_device!();
        let _: Tensor0D<_> = <Tensor1D<5, _> as MaxTo<_, _>>::max(dev.zeros());

        let _: Tensor1D<3, _> = <Tensor2D<5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor1D<5, _> = <Tensor2D<5, 3, _> as MaxTo<_, _>>::max(dev.zeros());

        let _: Tensor2D<5, 3, _> = <Tensor3D<7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor2D<7, 3, _> = <Tensor3D<7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor2D<7, 5, _> = <Tensor3D<7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());

        let _: Tensor3D<7, 5, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor3D<9, 5, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor3D<9, 7, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
        let _: Tensor3D<9, 7, 5, _> = <Tensor4D<9, 7, 5, 3, _> as MaxTo<_, _>>::max(dev.zeros());
    }

    #[test]
    fn test_max_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 2.0], [3.0, -2.0, 2.0]]);
        let r: Tensor1D<3, _, _> = t.trace().max();
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
        let r: Tensor1D<2, _, _> = t.trace().max();
        assert_eq!(r.array(), [2.0, 3.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).array(), [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_max_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _> = dev.randn();
        let r: Tensor1D<4, _, _> = t.trace().max();
        let r2: Tensor1D<4, _, _> = MaxTo::<Tensor2D<3, 4, _, _>, _>::max(t.trace()).max();
        assert_close(&r.array(), &r2.array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).array(), &g2.get(&t).array());
    }
}
