use crate::{
    arrays::{Dtype, HasShape, Shape},
    devices::{
        device::{HasErr, UnaryKernel},
        unary_ops, Device,
    },
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

/// Broadcast self into `T` along `Axes`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// // broadcast axis 1
/// let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast();
///
/// // broadcast axes 0, 1
/// let _: Tensor3D<7, 5, 3> = Tensor1D::<3>::zeros().broadcast();
///
/// // broadcast axes 1, 2, 3
/// let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast();
/// ```
pub trait BroadcastTo<T: HasShape, Axes>: HasErr {
    fn broadcast(self) -> T
    where
        T::Shape: Default,
    {
        self.try_broadcast().unwrap()
    }
    fn try_broadcast(self) -> Result<T, Self::Err>
    where
        T::Shape: Default,
    {
        self.try_broadcast_to(&Default::default())
    }
    fn broadcast_to(self, dst: &T::Shape) -> T {
        self.try_broadcast_to(dst).unwrap()
    }
    fn try_broadcast_to(self, dst: &T::Shape) -> Result<T, Self::Err>;
}

impl<Src: Shape, Dst: Shape, Axes: 'static + Copy, E: Dtype, D: Device, T: Tape<D>>
    BroadcastTo<Tensor<Dst, E, D, T>, Axes> for Tensor<Src, E, D, T>
where
    D: UnaryKernel<unary_ops::Broadcast<Dst, Axes>, Src, Dst, E>,
{
    fn try_broadcast_to(self, dst: &Dst) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let op = dst.into();
        try_unary_op(op, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::Zeros;
    use crate::tensor::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_valid_1d_broadcasts() {
        let dev = build_test_device!();

        let _: Tensor1D<5, _> = Zeros::<Tensor0D<_>>::zeros(&dev).broadcast();

        let _: Tensor2D<5, 3, _> = Zeros::<Tensor1D<3, _>>::zeros(&dev).broadcast();
        let _: Tensor2D<5, 3, _> = Zeros::<Tensor1D<5, _>>::zeros(&dev).broadcast();

        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor2D<5, 7, _>>::zeros(&dev).broadcast();
        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor2D<3, 7, _>>::zeros(&dev).broadcast();
        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor2D<3, 5, _>>::zeros(&dev).broadcast();
        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor2D<3, 5, _>>::zeros(&dev).broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor3D<5, 7, 9, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor3D<3, 7, 9, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor3D<3, 5, 9, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor3D<3, 5, 7, _>>::zeros(&dev).broadcast();
    }

    #[test]
    fn test_valid_2d_broadcasts() {
        let dev = build_test_device!();

        let _: Tensor2D<5, 3, _> = Zeros::<Tensor0D<_>>::zeros(&dev).broadcast();

        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor1D<3, _>>::zeros(&dev).broadcast();
        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor1D<5, _>>::zeros(&dev).broadcast();
        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor1D<7, _>>::zeros(&dev).broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<3, 5, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<3, 7, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<3, 9, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<5, 7, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<5, 9, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor2D<7, 9, _>>::zeros(&dev).broadcast();
    }

    #[test]
    fn test_valid_3d_broadcasts() {
        let dev = build_test_device!();

        let _: Tensor3D<3, 5, 7, _> = Zeros::<Tensor0D<_>>::zeros(&dev).broadcast();

        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor1D<3, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor1D<5, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor1D<7, _>>::zeros(&dev).broadcast();
        let _: Tensor4D<3, 5, 7, 9, _> = Zeros::<Tensor1D<9, _>>::zeros(&dev).broadcast();
    }

    // #[test]
    // fn test_broadcast_backwards() {
    //     let dev = build_test_device!();
    //     let a: Tensor1D<3, _> = dev.randn();
    //     let b: Tensor2D<5, 3, _> = dev.randn();
    //     let a_up: Tensor2D<5, 3, _, OwnedTape<_>> = a.trace().broadcast();
    //     a_up.data().assert_close(&[*a.data(); 5], 1e-4);
    //     let r = mul(a_up, b.clone());
    //     let g = backward(r.exp().mean());
    //     // a's gradient: (b * (b * a).exp()).sum(0) / 15
    //     // b's gradient: (a * (b * a).exp()) / 15
    //     let a_up: Tensor2D<5, 3> = a.clone().broadcast();
    //     let a_grad = mul(mul(b.clone(), a_up.clone()).exp(), b.clone()).sum::<_, Axis<0>>() / 15.0;
    //     let b_grad = mul(mul(b.clone(), a_up.clone()).exp(), a_up) / 15.0;
    //     g.ref_gradient(&a).assert_close(a_grad.data(), 1e-4);
    //     g.ref_gradient(&b).assert_close(b_grad.data(), 1e-4);
    // }
}
