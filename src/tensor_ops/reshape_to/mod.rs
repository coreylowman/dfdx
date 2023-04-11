mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

pub trait ReshapeKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>;
    fn backward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

/// Changes the shape of a tensor without re-ordering axes. If the tensor is contiguous
/// already, then no data movement will occur. If the tensor is not contiguous, the
/// result of this will be contiguous.
///
/// Compile time reshapes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank2<2, 4>, f32, _> = dev.zeros();
/// let t: Tensor<Rank1<8>, f32, _> = t.reshape();
/// ```
///
/// Compile time failure:
/// ```compile_fail
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank2<2, 4>, f32, _> = dev.zeros();
/// let t: Tensor<Rank1<7>, f32, _> = t.reshape();
/// ```
///
/// Runtime reshapes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank2<2, 4>, f32, _> = dev.zeros();
/// let t: Tensor<(usize, ), f32, _> = t.reshape_like(&(8, )).unwrap();
/// ```
pub trait ReshapeTo: HasErr + HasShape {
    /// Reshapes a tensor to a different compile time shape.
    fn reshape<Dst: ConstShape>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: ConstShape,
    {
        <Self::Shape as AssertSameNumel<Dst>>::assert_same_numel();
        self.try_reshape().unwrap()
    }
    /// Reshapes a tensor to a different compile time shape.
    fn try_reshape<Dst: ConstShape>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: ConstShape,
    {
        <Self::Shape as AssertSameNumel<Dst>>::assert_same_numel();
        self.try_reshape_like::<Dst>(&Default::default()).unwrap()
    }
    /// Reshapes a tensor to a different runtime shape.
    fn reshape_like<Dst: Shape>(self, dst: &Dst) -> Option<Self::WithShape<Dst>> {
        self.try_reshape_like(dst).map(Result::unwrap)
    }
    /// Ensures the tensor's memory is contiguous.
    ///
    /// If the memory is already contiguous no copying is performed.
    fn contiguous(self) -> Self::WithShape<Self::Shape> {
        self.try_contiguous().unwrap()
    }
    /// See [`ReshapeTo::contiguous`]
    fn try_contiguous(self) -> Result<Self::WithShape<Self::Shape>, Self::Err> {
        let shape = *self.shape();
        self.try_reshape_like(&shape).unwrap()
    }
    /// Reshapes a tensor to a different runtime shape.
    fn try_reshape_like<Dst: Shape>(
        self,
        dst: &Dst,
    ) -> Option<Result<Self::WithShape<Dst>, Self::Err>>;
}

impl<S: Shape, E: Dtype, D: ReshapeKernel<E>, T: Tape<E, D>> ReshapeTo for Tensor<S, E, D, T> {
    fn try_reshape_like<Dst: Shape>(
        self,
        dst: &Dst,
    ) -> Option<Result<Self::WithShape<Dst>, Self::Err>> {
        (self.shape().num_elements() == dst.shape().num_elements()).then(|| {
            if self.shape.strides() == self.strides {
                Ok(Tensor {
                    id: self.id,
                    data: self.data,
                    shape: *dst,
                    strides: dst.strides(),
                    device: self.device,
                    tape: self.tape,
                })
            } else {
                let (inp, mut tape) = self.split_tape();
                let out = inp.device.forward(dst, &inp)?;
                let inp_ghost = inp.ghost();
                let out_ghost = out.ghost();
                let dst = *dst;
                tape.add_backward_op(move |grads| {
                    grads.try_alloc_for(&inp_ghost)?;
                    grads.try_alloc_for(&out_ghost)?;
                    let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
                    inp.device.backward(&dst, &inp, grad_inp, grad_out)
                });
                Ok(out.put_tape(tape))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    use super::*;

    #[test]
    fn test_invalid_reshape() {
        let dev: TestDevice = Default::default();
        let t: Tensor<(usize,), TestDtype, _> = dev.zeros_like(&(5,));
        assert!(t.reshape_like(&(7,)).is_none());
    }

    #[test]
    fn test_unit_non_contiguous_reshapes() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank2<2, 3>, usize, _> = dev.zeros::<Rank0>().broadcast();
        let _: Tensor<Rank1<6>, usize, _> = t.reshape();
    }

    #[test]
    fn test_valid_reshapes() {
        let dev: TestDevice = Default::default();

        let t: Tensor<Rank0, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<1, 1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<1, 1, 1>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<1, 1, 1, 1>, TestDtype, _> = t.reshape();

        let t: Tensor<Rank1<16>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.reshape();

        let t: Tensor<Rank2<2, 8>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<8, 2>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.reshape();

        let t: Tensor<Rank3<2, 2, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<4, 2, 2>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = t.reshape();

        let t: Tensor<Rank4<2, 2, 2, 2>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank1<16>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank2<2, 8>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank3<2, 2, 4>, TestDtype, _> = t.clone().reshape();
        let _: Tensor<Rank4<4, 1, 2, 2>, TestDtype, _> = t.reshape();
    }

    #[test]
    fn test_1d_reshape() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b = a.leaky_trace().reshape::<Rank2<2, 3>>();
        assert_eq!(b.array(), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let g = b.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865,
            ],
        )
    }

    #[test]
    fn test_1d_reshape_non_contiguous() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let b = a
            .leaky_trace()
            .permute::<Rank2<3, 2>, _>()
            .reshape::<Rank1<6>>();
        assert_eq!(b.array(), [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]);
        let g = b.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                [0.18419516, 0.20356713, 0.22497648],
                [0.24863747, 0.2747869, 0.3036865],
            ],
        )
    }

    #[test]
    fn test_reshape_broadcasted() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank2<2, 3>, TestDtype, _> = dev.tensor([1., 2., 3.]).broadcast();
        let b: Tensor<Rank2<3, 2>, TestDtype, _> = a.clone().reshape();

        #[cfg(feature = "cuda")]
        use cudarc::driver::DeviceSlice;

        assert_eq!(b.data.len(), 6);
        assert_eq!(a.as_vec(), b.as_vec());
        assert_eq!(b.array(), [[1., 2.], [3., 1.], [2., 3.]]);
    }

    #[test]
    fn test_contiguous() {
        let dev: TestDevice = Default::default();

        let a: Tensor<_, TestDtype, _> = dev.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);

        let b1 = a.clone().contiguous();
        assert_eq!(a.strides, b1.strides);

        let b2: Tensor<_, TestDtype, _> = a.permute::<Rank2<3, 2>, _>().contiguous();
        assert_eq!(b2.strides, [2, 1]);
    }
}
