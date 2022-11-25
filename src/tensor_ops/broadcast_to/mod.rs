mod cpu_kernel;

use crate::{
    arrays::{Dtype, HasShape, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

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

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct BroadcastKernelOp<S, Axes>(pub(crate) S, std::marker::PhantomData<Axes>);
impl<S: Copy, Axes> From<&S> for BroadcastKernelOp<S, Axes> {
    fn from(s: &S) -> Self {
        Self(*s, std::marker::PhantomData)
    }
}

impl<Src: Shape, Dst: Shape, Axes: 'static + Clone, E: Dtype, D: Device, T: Tape<D>>
    BroadcastTo<Tensor<Dst, E, D, T>, Axes> for Tensor<Src, E, D, T>
where
    D: UnaryKernel<BroadcastKernelOp<Dst, Axes>, Src, Dst, E>,
{
    fn try_broadcast_to(self, dst: &Dst) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        try_unary_op(dst.into(), self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AsArray, Randn, Zeros};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{build_test_device, AssertClose};

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

    #[test]
    fn test_broadcast_backwards() {
        let dev = build_test_device!();
        let a: Tensor1D<3, _> = dev.randn();
        let b: Tensor2D<5, 3, _> = dev.randn();
        let a_up: Tensor2D<5, 3, _, _> = a.trace().broadcast();
        a_up.as_array().assert_close(&[a.as_array(); 5], 1e-4);
        let r = a_up * b.clone();
        let g = r.exp().mean().backward();

        let a_up: Tensor2D<5, 3, _> = a.clone().broadcast();
        // a's gradient: (b * (b * a).exp()).sum(0) / 15
        let a_grad: Tensor1D<3, _> = (b.clone() * (b.clone() * a_up.clone()).exp()).sum() / 15.0;
        // b's gradient: (a * (b * a).exp()) / 15
        let b_grad = (a_up.clone() * (b.clone() * a_up).exp()) / 15.0;
        g.get(&a).as_array().assert_close(&a_grad.as_array(), 1e-4);
        g.get(&b).as_array().assert_close(&b_grad.as_array(), 1e-4);
    }
}
