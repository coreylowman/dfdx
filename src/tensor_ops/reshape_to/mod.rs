mod cpu_kernel;

use crate::{gradients::Tape, shapes::*, tensor::storage::*, tensor::*};

pub trait ReshapeKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>;
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>;
}

/// **Requires Nightly** Reshape `Self` into `T`.
pub trait ReshapeTo: HasErr + HasShape {
    fn reshape<Dst: Shape + Default>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: HasSameNumelAs<Dst>,
    {
        self.try_reshape().unwrap()
    }
    fn try_reshape<Dst: Shape + Default>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasSameNumelAs<Dst>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage + ReshapeKernel<E>, T: Tape<D>> ReshapeTo
    for Tensor<S, E, D, T>
{
    fn try_reshape<Dst: Shape + Default>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasSameNumelAs<Dst>,
    {
        let dst: Dst = Default::default();
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward(dst, &inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(grad_inp, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    use super::*;

    #[test]
    fn test_valid_reshapes() {
        let dev = build_test_device!();

        let t: Tensor0D<_> = dev.zeros();
        let _: Tensor1D<1, _> = t.clone().reshape();
        let _: Tensor2D<1, 1, _> = t.clone().reshape();
        let _: Tensor3D<1, 1, 1, _> = t.clone().reshape();
        let _: Tensor4D<1, 1, 1, 1, _> = t.clone().reshape();

        let t: Tensor1D<16, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor2D<2, 8, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<8, 2, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor3D<2, 2, 4, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<4, 2, 2, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor4D<2, 2, 2, 2, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<4, 1, 2, 2, _> = t.clone().reshape();
    }

    #[test]
    fn test_1d_reshape() {
        let dev = build_test_device!();
        let a = dev.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b = a.trace().reshape::<Rank2<2, 3>>();
        assert_eq!(b.array(), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let g = b.exp().mean().backward();
        assert_eq!(
            g.get(&a).array(),
            [0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
