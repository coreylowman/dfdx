mod cpu_kernel;

use crate::{gradients::Tape, shapes::*, tensor::*};

pub trait BroadcastKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>;
}

/// Broadcast self into a new shape.
pub trait BroadcastTo: HasErr + HasShape {
    /// Broadcast into shape `Dst` along axes `Ax`:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 7>, f32> = dev.zeros();
    ///
    /// // broadcast axis 1
    /// let _ = a.clone().broadcast::<Rank3<3, 5, 7>, _>();
    ///
    /// // broadcast axis 0 and axis 2
    /// let _ = a.clone().broadcast::<Rank4<1, 3, 5, 7>, _>();
    /// ```
    fn broadcast<Dst: Shape + Default, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(&Default::default()).unwrap()
    }
    /// Fallible version of [BroadcastTo::broadcast]
    fn try_broadcast<Dst: Shape + Default, Ax: Axes>(
        self,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(&Default::default())
    }
    /// Same as [BroadcastTo::broadcast], but the target shape is given
    fn broadcast_like<Dst: Shape, Ax: Axes>(self, dst: &Dst) -> Self::WithShape<Dst>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        self.try_broadcast_like(dst).unwrap()
    }
    /// fallible version of [BroadcastTo::broadcast_like]
    fn try_broadcast_like<Dst: Shape, Ax: Axes>(
        self,
        dst: &Dst,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: BroadcastKernel<E>, T: Tape<D>> BroadcastTo for Tensor<S, E, D, T> {
    fn try_broadcast_like<Dst: Shape, Ax: Axes>(
        self,
        dst: &Dst,
    ) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: BroadcastShapeTo<Dst, Ax>,
    {
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward(*dst, &inp.storage)?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{AssertClose, TestDevice};

    #[test]
    fn test_valid_1d_broadcasts() {
        let dev: TestDevice = Default::default();

        let _ = dev.rand::<Rank0>().broadcast::<Rank1<5>, _>();

        let _ = dev.rand::<Rank1<3>>().broadcast::<Rank2<5, 3>, _>();
        let _ = dev.rand::<Rank1<5>>().broadcast::<Rank2<5, 3>, _>();

        let _ = dev.rand::<Rank2<5, 7>>().broadcast::<Rank3<3, 5, 7>, _>();
        let _ = dev.rand::<Rank2<3, 7>>().broadcast::<Rank3<3, 5, 7>, _>();
        let _ = dev.rand::<Rank2<3, 5>>().broadcast::<Rank3<3, 5, 7>, _>();
        let _ = dev.rand::<Rank2<3, 5>>().broadcast::<Rank3<3, 5, 7>, _>();

        let _ = dev
            .rand::<Rank3<5, 7, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>, _>();
        let _ = dev
            .rand::<Rank3<3, 7, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>, _>();
        let _ = dev
            .rand::<Rank3<3, 5, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>, _>();
        let _ = dev
            .rand::<Rank3<3, 5, 7>>()
            .broadcast::<Rank4<3, 5, 7, 9>, _>();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let dev: TestDevice = Default::default();

        let _: Tensor2D<5, 3, _> = dev.zeros::<Rank0>().broadcast();

        let _: Tensor3D<3, 5, 7, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor3D<3, 5, 7, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor3D<3, 5, 7, _> = dev.zeros::<Rank1<7>>().broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<3, 5>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<3, 7>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<3, 9>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<5, 7>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<5, 9>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank2<7, 9>>().broadcast();
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let dev: TestDevice = Default::default();

        let _: Tensor3D<3, 5, 7, _> = dev.zeros::<Rank0>().broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<7>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<9>>().broadcast();
    }

    #[test]
    fn test_broadcast_backwards() {
        let dev: TestDevice = Default::default();
        let a = dev.randn::<Rank1<3>>();
        let b = dev.randn::<Rank2<5, 3>>();
        let a_up = a.trace().broadcast::<Rank2<5, 3>, _>();
        a_up.array().assert_close(&[a.array(); 5], 1e-4);
        let r = a_up * b.clone();
        let g = r.exp().mean().backward();

        let a_up = a.clone().broadcast::<Rank2<5, 3>, _>();
        // a's gradient: (b * (b * a).exp()).sum(0) / 15
        let a_grad = (b.clone() * (b.clone() * a_up.clone()).exp()).sum::<Rank1<3>, _>() / 15.0;
        // b's gradient: (a * (b * a).exp()) / 15
        let b_grad = (a_up.clone() * (b.clone() * a_up).exp()) / 15.0;
        g.get(&a).array().assert_close(&a_grad.array(), 1e-4);
        g.get(&b).array().assert_close(&b_grad.array(), 1e-4);
    }
}
