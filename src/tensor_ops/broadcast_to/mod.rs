mod cpu_kernel;

use crate::{arrays::*, gradients::Tape, tensor::*};

use super::Device;

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

/// Broadcast self into `T` along `Axes`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// // broadcast axis 1
/// let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast_into();
///
/// // broadcast axes 0, 1
/// let _: Tensor3D<7, 5, 3> = Tensor1D::<3>::zeros().broadcast_into();
///
/// // broadcast axes 1, 2, 3
/// let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast_into();
/// ```
pub trait BroadcastTo<T: HasShape, Ax>: HasErr {
    fn broadcast_to(self, dst: &T::Shape) -> T {
        self.try_broadcast_to(dst).unwrap()
    }
    fn try_broadcast_to(self, dst: &T::Shape) -> Result<T, Self::Err>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape, Ax: Axes>
    BroadcastTo<Tensor<Dst, E, D, T>, Ax> for Tensor<S, E, D, T>
where
    D: BroadcastKernel<E>,
    S: BroadcastShapeTo<Dst, Ax>,
{
    fn try_broadcast_to(self, dst: &Dst) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(*dst, &inp.storage)?;
        let out = inp.device.upgrade(storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device.backward(grad_inp, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

pub trait Broadcast<Ax: Axes>: HasErr + HasShape {
    fn broadcast<Dst: Shape + Default>(self) -> Self::WithShape<Dst>
    where
        Self: BroadcastTo<Self::WithShape<Dst>, Ax>,
    {
        self.broadcast_to(&Default::default())
    }

    fn try_broadcast<Dst: Shape + Default>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self: BroadcastTo<Self::WithShape<Dst>, Ax>,
    {
        self.try_broadcast_to(&Default::default())
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Ax: Axes> Broadcast<Ax> for Tensor<S, E, D, T> {}

pub trait BroadcastAlong<Dst: Shape>: HasShape + HasErr {
    fn broadcast_along<Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self: BroadcastTo<Self::WithShape<Dst>, Ax>,
        Dst: Default,
    {
        self.broadcast_to(&Default::default())
    }

    fn try_broadcast_along<Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self: BroadcastTo<Self::WithShape<Dst>, Ax>,
        Dst: Default,
    {
        self.try_broadcast_to(&Default::default())
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>, Dst: Shape> BroadcastAlong<Dst>
    for Tensor<S, E, D, T>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::{build_test_device, AssertClose};

    #[test]
    fn test_valid_1d_broadcasts() {
        let dev = build_test_device!();

        let _ = dev.zeros::<Rank0>().broadcast::<Rank1<5>>();

        let _ = dev.zeros::<Rank1<3>>().broadcast::<Rank2<5, 3>>();
        let _ = dev.zeros::<Rank1<5>>().broadcast::<Rank2<5, 3>>();

        let _ = dev.zeros::<Rank2<5, 7>>().broadcast::<Rank3<3, 5, 7>>();
        let _ = dev.zeros::<Rank2<3, 7>>().broadcast::<Rank3<3, 5, 7>>();
        let _ = dev.zeros::<Rank2<3, 5>>().broadcast::<Rank3<3, 5, 7>>();
        let _ = dev.zeros::<Rank2<3, 5>>().broadcast::<Rank3<3, 5, 7>>();

        let _ = dev
            .zeros::<Rank3<5, 7, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>>();
        let _ = dev
            .zeros::<Rank3<3, 7, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>>();
        let _ = dev
            .zeros::<Rank3<3, 5, 9>>()
            .broadcast::<Rank4<3, 5, 7, 9>>();
        let _ = dev
            .zeros::<Rank3<3, 5, 7>>()
            .broadcast::<Rank4<3, 5, 7, 9>>();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let dev = build_test_device!();

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
        let dev = build_test_device!();

        let _: Tensor3D<3, 5, 7, _> = dev.zeros::<Rank0>().broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<3>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<5>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<7>>().broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = dev.zeros::<Rank1<9>>().broadcast();
    }

    #[test]
    fn test_broadcast_backwards() {
        let dev = build_test_device!();
        let a = dev.randn::<Rank1<3>>();
        let b = dev.randn::<Rank2<5, 3>>();
        let a_up = a.trace().broadcast::<Rank2<5, 3>>();
        a_up.array().assert_close(&[a.array(); 5], 1e-4);
        let r = a_up * b.clone();
        let g = r.exp().mean().backward();

        let a_up = a.clone().broadcast::<Rank2<5, 3>>();
        // a's gradient: (b * (b * a).exp()).sum(0) / 15
        let a_grad = (b.clone() * (b.clone() * a_up.clone()).exp()).sum_to::<Rank1<3>>() / 15.0;
        // b's gradient: (a * (b * a).exp()) / 15
        let b_grad = (a_up.clone() * (b.clone() * a_up).exp()) / 15.0;
        g.get(&a).array().assert_close(&a_grad.array(), 1e-4);
        g.get(&b).array().assert_close(&b_grad.array(), 1e-4);
    }
}
