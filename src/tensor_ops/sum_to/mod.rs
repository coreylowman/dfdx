mod cpu_kernel;

use super::device::Device;
use crate::{arrays::*, devices::*, gradients::Tape, tensor::*};

pub trait SumKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Dst: BroadcastShapeTo<Src, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Dst: BroadcastShapeTo<Src, Ax>;
}

/// Sum values along axes `Axes` of `T`.
///
/// **Pytorch equivalent**: `t.sum(Axes)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor0D = t.sum();
/// ```
///
/// Reducing 1 axis:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor2D<3, 4> = t.sum();
/// ```
///
/// Reducing multiple axes:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 4> = TensorCreator::zeros();
/// let _: Tensor1D<4> = t.sum();
/// ```
///
/// Specifying axes instead of output types:
/// ```rust
/// todo!()
/// ```
pub trait SumTo<T, Axes>: HasErr {
    fn sum(self) -> T {
        self.try_sum().unwrap()
    }
    fn try_sum(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>, Dst: Shape + Default, Ax: Axes>
    SumTo<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    D: SumKernel<E>,
    Dst: BroadcastShapeTo<Src, Ax>,
{
    fn try_sum(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let dst: Dst = Default::default();
        let (inp, mut tape) = self.split_tape();
        let storage = inp.device.forward(dst, &inp.storage)?;
        let out = make_tensor(&inp.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            inp.device.backward(grad_inp, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn sum_along<Ax: Axes>(self) -> Tensor<S::Reduced, E, D, T>
    where
        S: ReduceShape<Ax>,
    {
        self.try_sum_along().unwrap()
    }

    pub fn try_sum_along<Ax: Axes>(self) -> Result<Tensor<S::Reduced, E, D, T>, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::AsArray;
    use crate::devices::Randn;
    use crate::gradients::OwnedTape;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_sum_1d() {
        let dev = build_test_device!();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r: Tensor0D<_, OwnedTape<_>> = t.trace().sum();
        assert_eq!(r.as_array(), 6.0);
        // NOTE: .exp() to make sure its using result grad properly
        let g = r.exp().backward();
        assert_eq!(g.get(&t).as_array(), [403.4288; 3]);
    }

    #[test]
    fn test_sum_axis_0_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r: Tensor1D<3, _, _> = t.trace().sum();
        assert_eq!(r.as_array(), [-1.0, 6.0, -3.0]);
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.12262648, 134.47627, 0.01659569]; 2]
        );
    }

    #[test]
    fn test_sum_axis_1_2d() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, -6.0]]);
        let r: Tensor1D<2, _, _> = t.trace().sum();
        assert_eq!(r.as_array(), [6.0, -4.0]);
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).as_array(), [[201.7144; 3], [0.00915782; 3]]);
    }

    #[test]
    fn test_sum_axes_3d_to_1d() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _, _> = dev.randn();
        let r: Tensor1D<3, _, _> = t.trace().sum();
        let a: Tensor2D<3, 4, _, _> = t.trace().sum();
        let r2: Tensor1D<3, _, _> = a.sum();
        assert_close(&r.as_array(), &r2.as_array());
        let g = r.mean().backward();
        let g2 = r2.mean().backward();
        assert_close(&g.get(&t).as_array(), &g2.get(&t).as_array());
    }
}
