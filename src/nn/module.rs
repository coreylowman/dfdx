use crate::{optim::GradientUpdate, prelude::Cpu, shapes::Dtype, tensor_ops::Device};

/// Immutable forward of `Input` that produces [Module::Output].
/// See [ModuleMut] for mutable forward.
pub trait Module<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Forward `Input` through the module and produce [Module::Output].
    ///
    /// **See [ModuleMut::forward_mut()] for version that can mutate `self`.**
    fn forward(&self, input: Input) -> Self::Output;
}

/// Mutable forward of `Input` that produces [ModuleMut::Output].
/// See [Module] for immutable forward.
pub trait ModuleMut<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Forward `Input` through the module and produce [ModuleMut::Output].
    ///
    /// **See [Module::forward()] for immutable version**
    fn forward_mut(&mut self, input: Input) -> Self::Output;
}

/// Something that can reset it's parameters.
pub trait ResetParams<D: Device<E>, E: Dtype>: Sized {
    /// Construct it on the device
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }
    /// Fallible version of [ResetParams::build]
    fn try_build(device: &D) -> Result<Self, D::Err>;

    /// Mutates parameters. Each implementor
    /// of this trait decides how the parameters are initialized. In
    /// fact, some impls may not even use randomness.
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }

    /// Fallible version of [ResetParams::reset_params].
    fn try_reset_params(&mut self) -> Result<(), D::Err>;
}

/// Extension trait for [Device] that can build anything that implements [ResetParams].
pub trait ModuleBuilder<E: Dtype>: Device<E> {
    fn build_module<M: ResetParams<Self, E>>(&self) -> M {
        ResetParams::build(self)
    }
    fn try_build<M: ResetParams<Self, E>>(&self) -> Result<M, Self::Err> {
        ResetParams::try_build(self)
    }
}
impl<D: Device<E>, E: Dtype> ModuleBuilder<E> for D {}

/// Marker trait for modules with no updatable parameters. These have
/// blanket impls for [ResetParams], [GradientUpdate], and [ModuleMut]
pub trait ZeroSizedModule: Default {}

impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> ResetParams<D, E> for T {
    fn try_build(_: &D) -> Result<Self, <D>::Err> {
        Ok(Default::default())
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        Ok(())
    }
}

impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> GradientUpdate<D, E> for T {
    fn update<U>(&mut self, _: &mut U, _: &mut crate::optim::UnusedTensors) -> Result<(), <D>::Err>
    where
        U: crate::optim::ParamUpdater<D, E>,
    {
        Ok(())
    }
}

/// Marker trait for modules that don't have different behavior between
/// mutable forwards and non-mutable forwards
pub trait NonMutableModule {}

impl<M: NonMutableModule, T> ModuleMut<T> for M
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

/// A trait which allows a module to be used with the [OnDevice] type alias.
/// Here's an example of how this can be implemented for a custom struct:
/// ```rust
/// use dfdx::prelude::*;
///
/// struct MLP<D: Device<f32>> {
///     l1: Linear<5, 10, D>,
///     a1: ReLU,
///     l2: Linear<10, 1, D>,
/// }
/// 
/// // Need two device types to allow converting from one device to another
/// impl<D1: Device<f32>, D2: Device<f32>> OnDeviceTrait<D2> for MLP<D1> {
///     type Output = MLP<D2>;
/// }
/// ````
pub trait OnDeviceTrait<D> {
    type Output;
}

/// A type alias that allows types that implement [OnDeviceTrait] to be changed to a corresponding
/// type on the specified device.
/// Examples:
/// ```rust
/// # use dfdx::nn::*;
/// type MLP<D> = OnDevice<(Linear<5, 10>, ReLU, Linear<10, 1>), D>;
/// ```
///
/// ```rust
/// # // Only compiles with cuda
/// # use dfdx::prelude::*;
/// #
/// // All modules exist on the cpu by default
/// type CpuMLP = (Linear<5, 10>, ReLU, Linear<10, 1>);
/// type MLP<D> = OnDevice<CpuMLP, D>;
/// type CudaMLP = OnDevice<CpuMLP, Cuda>;
/// ```
pub type OnDevice<M, D> = <M as OnDeviceTrait<D>>::Output;

#[cfg(feature = "cuda")]
pub type OnCuda<M> = OnDevice<M, crate::prelude::Cuda>;
pub type OnCpu<M> = OnDevice<M, Cpu>;

impl<T: ZeroSizedModule, D> OnDeviceTrait<D> for T {
    type Output = T;
}
