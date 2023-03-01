mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{gradients::Tape, shapes::*, tensor::*};

pub trait ReshapeKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>;
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

/// **Requires Nightly** Change the shape of a tensor moving data around.
pub trait ReshapeTo: HasErr + HasShape {
    fn reshape<Dst: ConstShape>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: HasSameNumelAs<Dst>,
    {
        self.try_reshape().unwrap()
    }
    fn try_reshape<Dst: ConstShape>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: HasSameNumelAs<Dst>,
    {
        self.try_reshape_like::<Dst>(&Default::default())
    }
    fn reshape_like<Dst: Shape>(self, dst: &Dst) -> Self::WithShape<Dst> {
        self.try_reshape_like(dst).unwrap()
    }
    fn try_reshape_like<Dst: Shape>(self, dst: &Dst) -> Result<Self::WithShape<Dst>, Self::Err>;
}

impl<S: Shape, E: Dtype, D: ReshapeKernel<E>, T: Tape<E, D>> ReshapeTo for Tensor<S, E, D, T> {
    fn try_reshape_like<Dst: Shape>(self, dst: &Dst) -> Result<Self::WithShape<Dst>, Self::Err> {
        assert_eq!(self.shape().num_elements(), dst.shape().num_elements());
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(dst, &inp)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(&inp, grad_inp, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    use super::*;

    #[test]
    #[should_panic]
    fn test_invalid_reshape() {
        let dev: TestDevice = Default::default();
        let t: Tensor<(usize,), TestDtype, _> = dev.zeros_like(&(5,));
        let _ = t.reshape_like(&(7,));
    }

    #[test]
    fn test_valid_reshapes() {
        let dev: TestDevice = Default::default();

        let t: Tensor<Rank0, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<1, 1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<1, 1, 1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<1, 1, 1, 1>, TestDtype, _> = t.clone().reshape();

        let t: Tensor<Rank1<16>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.clone().reshape();

        let t: Tensor<Rank2<2, 8>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<8, 2>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.clone().reshape();

        let t: Tensor<Rank3<2, 2, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<4, 2, 2>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.clone().reshape();

        let t: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<4, 1, 2, 2>, TestDtype, _> = t.clone().reshape();
    }

    #[test]
    fn test_1d_reshape() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b = a.trace().reshape::<Rank2<2, 3>>();
        assert_eq!(b.array(), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let g = b.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865,
            ],
        )
    }
}
