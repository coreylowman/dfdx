use crate::{optim::GradientUpdate, shapes::Dtype, tensor_ops::Device};

#[cfg(feature = "cuda")]
pub use crate::tensor::OnCuda;
pub use crate::tensor::{OnCpu, OnDevice, ToDevice};

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

/// Something that can be built. Related to [BuildOnDevice]
pub trait BuildModule<D: Device<E>, E: Dtype>: Sized {
    /// Construct it on the device
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }
    /// Fallible version of [ResetParams::build]
    fn try_build(device: &D) -> Result<Self, D::Err>;
}

/// Something that can be built on a different device
/// than it is on. Builds [BuildOnDevice::Built].
///
/// Related to [BuildModule]
pub trait BuildOnDevice<D: Device<E>, E: Dtype> {
    type Built: BuildModule<D, E>;
    fn build_on_device(device: &D) -> Self::Built {
        BuildModule::build(device)
    }
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        BuildModule::try_build(device)
    }
}

/// Something that can reset it's parameters.
pub trait ResetParams<D: Device<E>, E: Dtype>: Sized {
    /// Mutates parameters. Each implementor
    /// of this trait decides how the parameters are initialized. In
    /// fact, some impls may not even use randomness.
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }

    /// Fallible version of [ResetParams::reset_params].
    fn try_reset_params(&mut self) -> Result<(), D::Err>;
}

/// Marker trait for modules with no updatable parameters. These have
/// blanket impls for [ResetParams], [GradientUpdate], and [ModuleMut]
pub trait ZeroSizedModule: Default {}

impl<T: ZeroSizedModule + Clone, D: Device<E>, E: Dtype> ResetParams<D, E> for T {
    fn try_build(_: &D) -> Result<Self, <D>::Err> {
        Ok(Default::default())
    }
}
impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for T {
    type Built = Self;
}
impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> ResetParams<D, E> for T {
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

impl<T: ZeroSizedModule + Clone, D> ToDevice<D> for T {
    type Output = T;

    fn to_device(&self, _device: &D) -> Self {
        self.clone()
    }
}
