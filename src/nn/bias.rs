use crate::{
    arrays::{Const, Dim, HasDtype, HasShape, Rank1},
    gradients::Tape,
    tensor::{AllocGradOn, Cpu, Tensor},
    tensor_ops::{BroadcastTo, Device},
    unique_id::HasUniqueId,
};

use super::{Module, ModuleMut};

#[derive(Clone, Debug)]
pub struct Bias1D<const M: usize, D: Device<f32> = Cpu> {
    pub beta: Tensor<Rank1<M>, f32, D>,
}

impl<const M: usize, D: Device<f32>> std::ops::Deref for Bias1D<M, D> {
    type Target = Tensor<Rank1<M>, f32, D>;
    fn deref(&self) -> &Self::Target {
        &self.beta
    }
}

impl<const M: usize, D: Device<f32>> HasUniqueId for Bias1D<M, D> {
    fn id(&self) -> &crate::unique_id::UniqueId {
        self.beta.id()
    }
}

impl<const M: usize, D: Device<f32>> HasShape for Bias1D<M, D> {
    type Shape = Rank1<M>;
    fn shape(&self) -> &Self::Shape {
        self.beta.shape()
    }
}

impl<const M: usize, D: Device<f32>> HasDtype for Bias1D<M, D> {
    type Dtype = f32;
}

impl<const M: usize, D: Device<f32>> AllocGradOn<D> for Bias1D<M, D> {
    fn try_alloc_grad(
        &self,
    ) -> Result<<D as crate::tensor::DeviceStorage>::Storage<Self::Shape, Self::Dtype>, <D>::Err>
    {
        self.beta.try_alloc_grad()
    }
}

impl<const M: usize, D: Device<f32>, T: Tape<D>> Module<Tensor<Rank1<M>, f32, D, T>>
    for Bias1D<M, D>
{
    type Output = Tensor<Rank1<M>, f32, D, T>;
    fn forward(&self, input: Tensor<Rank1<M>, f32, D, T>) -> Self::Output {
        input + self.beta.clone()
    }
}

impl<B: Dim, const M: usize, D: Device<f32>, T: Tape<D>> Module<Tensor<(B, Const<M>), f32, D, T>>
    for Bias1D<M, D>
{
    type Output = Tensor<(B, Const<M>), f32, D, T>;
    fn forward(&self, input: Tensor<(B, Const<M>), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_to(input.shape()) + input
    }
}

impl<B: Dim, S: Dim, const M: usize, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(B, S, Const<M>), f32, D, T>> for Bias1D<M, D>
{
    type Output = Tensor<(B, S, Const<M>), f32, D, T>;
    fn forward(&self, input: Tensor<(B, S, Const<M>), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_to(input.shape()) + input
    }
}

impl<const C: usize, D: Device<f32>, T> ModuleMut<T> for Bias1D<C, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[derive(Clone, Debug)]
pub struct Bias2D<const C: usize, D: Device<f32> = Cpu> {
    pub beta: Tensor<Rank1<C>, f32, D>,
}

impl<const C: usize, D: Device<f32>> std::ops::Deref for Bias2D<C, D> {
    type Target = Tensor<Rank1<C>, f32, D>;
    fn deref(&self) -> &Self::Target {
        &self.beta
    }
}

impl<const C: usize, D: Device<f32>> HasUniqueId for Bias2D<C, D> {
    fn id(&self) -> &crate::unique_id::UniqueId {
        self.beta.id()
    }
}

impl<const C: usize, D: Device<f32>> HasShape for Bias2D<C, D> {
    type Shape = Rank1<C>;
    fn shape(&self) -> &Self::Shape {
        self.beta.shape()
    }
}

impl<const C: usize, D: Device<f32>> HasDtype for Bias2D<C, D> {
    type Dtype = f32;
}

impl<const C: usize, D: Device<f32>> AllocGradOn<D> for Bias2D<C, D> {
    fn try_alloc_grad(
        &self,
    ) -> Result<<D as crate::tensor::DeviceStorage>::Storage<Self::Shape, Self::Dtype>, <D>::Err>
    {
        self.beta.try_alloc_grad()
    }
}

impl<const C: usize, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(Const<C>, H, W), f32, D, T>> for Bias2D<C, D>
{
    type Output = Tensor<(Const<C>, H, W), f32, D, T>;
    fn forward(&self, input: Tensor<(Const<C>, H, W), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_to(input.shape()) + input
    }
}

impl<B: Dim, const C: usize, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(B, Const<C>, H, W), f32, D, T>> for Bias2D<C, D>
{
    type Output = Tensor<(B, Const<C>, H, W), f32, D, T>;
    fn forward(&self, input: Tensor<(B, Const<C>, H, W), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_to(input.shape()) + input
    }
}

impl<const C: usize, D: Device<f32>, T> ModuleMut<T> for Bias2D<C, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}
