use crate::{gradients::Tape, shapes::*, tensor::*, tensor_ops::*};

use super::traits::*;

pub mod builder {
    #[derive(Debug)]
    pub struct Bias2D<const CHAN: usize>;
}

impl<const C: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E> for builder::Bias2D<C>
where
    Bias2D<C, E, D>: BuildModule<D, E>,
{
    type Built = Bias2D<C, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// Adds a learnable 1d bias to 3d and 4d inputs. Can be used with `crate::nn::modules::Conv2D`
/// to create a Biased conv.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// const NUM_CHANS: usize = 5;
/// type Model = Bias2D<NUM_CHANS>;
/// let model = dev.build_module::<Model, f32>();
///
/// // 3d input
/// let x: Tensor<Rank3<NUM_CHANS, 2, 3>, f32, _> = dev.sample_normal();
/// model.forward(x);
///
/// // 4d input
/// let x: Tensor<Rank4<10, NUM_CHANS, 2, 3>, f32, _> = dev.sample_normal();
/// model.forward(x);
/// ```
#[derive(Clone, Debug)]
pub struct Bias2D<const C: usize, E: Dtype, D: DeviceStorage> {
    pub bias: Tensor<Rank1<C>, E, D>,
}

impl<const C: usize, E: Dtype, D: Device<E>> BuildModule<D, E> for Bias2D<C, E, D> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            bias: device.try_zeros()?,
        })
    }
}

impl<const C: usize, E: Dtype, D: DeviceStorage> NonMutableModule for Bias2D<C, E, D> {}

impl<const C: usize, E: Dtype, D1: Device<E>, D2: Device<E>> ToDevice<D2> for Bias2D<C, E, D1> {
    type Output = Bias2D<C, E, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        Bias2D {
            bias: self.bias.to_device(device),
        }
    }
}

impl<const C: usize, E: Dtype, D: Device<E>> TensorCollection<E, D> for Bias2D<C, E, D> {
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            "beta",
            |s| &s.bias,
            |s| &mut s.bias,
            TensorOptions::reset_to_zeros(),
        )
    }
}

impl<const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Const<C>, H, W), E, D, T>> for Bias2D<C, E, D>
{
    type Output = Tensor<(Const<C>, H, W), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<C>, H, W), E, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let s = *input.shape();
        input.try_add(self.bias.retaped::<T>().try_broadcast_like(&s)?)
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, Const<C>, H, W), E, D, T>> for Bias2D<C, E, D>
{
    type Output = Tensor<(B, Const<C>, H, W), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(B, Const<C>, H, W), E, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let s = *input.shape();
        input.try_add(self.bias.retaped::<T>().try_broadcast_like(&s)?)
    }
}
