#[cfg(feature = "cuda")]
pub use crate::tensor::OnCuda;
pub use crate::tensor::{DeviceStorage, OnCpu, OnDevice, ToDevice};
use crate::{
    shapes::Dtype,
    tensor::visitors::{ModuleWalker, TensorCollection},
};

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
pub trait BuildModule<D: DeviceStorage, E: Dtype>: Sized {
    /// Construct it on the device
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }
    /// Fallible version of [BuildModule::build]
    fn try_build(device: &D) -> Result<Self, D::Err>;
}

/// Something that can be built on a different device
/// than it is on. Builds [ToDevice::Output].
///
/// Related to [BuildModule]
pub trait BuildOnDevice<D: DeviceStorage, E: Dtype> {
    type Built: BuildModule<D, E>;
    fn build_on_device(device: &D) -> Self::Built {
        Self::try_build_on_device(device).unwrap()
    }
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

/// An extension trait that allows you to build a module with a device
/// method. Also allows easy specification of Dtype.
pub trait DeviceBuildExt: DeviceStorage {
    fn build_module<M: BuildOnDevice<Self, E>, E: Dtype>(&self) -> M::Built {
        M::build_on_device(self)
    }
    fn try_build_module<M: BuildOnDevice<Self, E>, E: Dtype>(&self) -> Result<M::Built, Self::Err> {
        M::try_build_on_device(self)
    }
}
impl<D: DeviceStorage> DeviceBuildExt for D {}

/// Marker trait for modules with no updatable parameters. These have
/// blanket impls for [ResetParams], [GradientUpdate], and [ModuleMut]
pub trait ZeroSizedModule: Default {}

impl<T: ZeroSizedModule + BuildModule<D, E>, D: DeviceStorage, E: Dtype> BuildOnDevice<D, E> for T {
    type Built = T;
}

impl<E: Dtype, D: DeviceStorage, T: ZeroSizedModule> TensorCollection<E, D> for T {
    fn iter_tensors<V: ModuleWalker<Self, E, D>>(_: &mut V) -> Result<(), V::Err> {
        Ok(())
    }
}

impl<T: ZeroSizedModule + Clone, D> ToDevice<D> for T {
    type Output = T;
    fn to_device(&self, _device: &D) -> Self {
        self.clone()
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
