mod cpu_kernel;

use crate::{arrays::*, devices::*, gradients::Tape, tensor::*};

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
pub trait MinTo<T, Ax>: HasErr {
    fn min(self) -> T {
        self.try_min().unwrap()
    }
    fn try_min(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape + Default, Ax: Axes>
    MinTo<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    D: MinReduceKernel<E>,
    Src: ReduceShapeTo<Dst, Ax>,
{
    fn try_min(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let dst: Dst = Default::default();
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(dst, &inp.storage)?;
        let out = make_tensor(&inp.device, storage);
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
    fn test_valids_min_axis() {
        let dev = build_test_device!();
        let _: Tensor0D<_> = <Tensor1D<5, _> as MinTo<_, _>>::min(dev.zeros());

        let _: Tensor1D<3, _> = <Tensor2D<5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor1D<5, _> = <Tensor2D<5, 3, _> as MinTo<_, _>>::min(dev.zeros());

        let _: Tensor2D<5, 3, _> = <Tensor3D<7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor2D<7, 3, _> = <Tensor3D<7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor2D<7, 5, _> = <Tensor3D<7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());

        let _: Tensor3D<7, 5, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor3D<9, 5, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor3D<9, 7, 3, _> = <Tensor4D<9, 7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
        let _: Tensor3D<9, 7, 5, _> = <Tensor4D<9, 7, 5, 3, _> as MinTo<_, _>>::min(dev.zeros());
    }

    #[test]
    fn test_min_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r: Tensor1D<3, _, _> = t.trace().min();
        assert_eq!(r.as_array(), [1.0, -2.0, 2.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.90609396, 0.0, 2.463019], [0.0, 0.04511176, 2.463019]]
        );
    }

    #[test]
    fn test_min_axis_1_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 1.0, 2.0], [3.0, -2.0, 2.0]]);
        let r: Tensor1D<2, _, _> = t.trace().min();
        assert_eq!(r.as_array(), [1.0, -2.0]);
        let g = r.sum().backward();
        assert_eq!(g.get(&t).as_array(), [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_min_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _> = dev.randn();
        let r: Tensor1D<4, _, _> = t.trace().min();
        let r2: Tensor1D<4, _, _> = MinTo::<Tensor2D<3, 4, _, _>, _>::min(t.trace()).min();
        assert_close(&r.as_array(), &r2.as_array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).as_array(), &g2.get(&t).as_array());
    }
}
