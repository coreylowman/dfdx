pub use super::build_module::BuildModule;
pub use super::to_device::*;
pub use crate::tensor::DeviceStorage;
use crate::{prelude::Device, shapes::Dtype};

use super::tensor_collection::*;

/// Immutable forward of `Input` that produces [Module::Output].
/// See [ModuleMut] for mutable forward.
pub trait Module<Input> {
    /// The type that this unit produces given `Input`.
    type Output;
    type Error: core::fmt::Debug;

    fn try_forward(&self, input: Input) -> Result<Self::Output, Self::Error>;

    /// Forward `Input` through the module and produce [Module::Output].
    ///
    /// **See [ModuleMut::forward_mut()] for version that can mutate `self`.**
    fn forward(&self, input: Input) -> Self::Output {
        self.try_forward(input).unwrap()
    }
}

/// Mutable forward of `Input` that produces [ModuleMut::Output].
/// See [Module] for immutable forward.
pub trait ModuleMut<Input> {
    /// The type that this unit produces given `Input`.
    type Output;
    type Error: core::fmt::Debug;

    fn try_forward_mut(&mut self, input: Input) -> Result<Self::Output, Self::Error>;

    /// Forward `Input` through the module and produce [ModuleMut::Output].
    ///
    /// **See [Module::forward()] for immutable version**
    fn forward_mut(&mut self, input: Input) -> Self::Output {
        self.try_forward_mut(input).unwrap()
    }
}

/// Something that can be built on a different device
/// than it is on.
///
/// Related to [BuildModule]
pub trait BuildOnDevice<D: Device<E>, E: Dtype> {
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
    fn build_module<M: BuildOnDevice<Self, E>, E: Dtype>(&self) -> M::Built
    where
        Self: Device<E>,
    {
        M::build_on_device(self)
    }
    fn try_build_module<M: BuildOnDevice<Self, E>, E: Dtype>(&self) -> Result<M::Built, Self::Err>
    where
        Self: Device<E>,
    {
        M::try_build_on_device(self)
    }
}
impl<D: DeviceStorage> DeviceBuildExt for D {}

/// Marker trait for modules with no updatable parameters. These have
/// blanket impls for, and [ModuleMut]
pub trait ZeroSizedModule: Default {}

impl<T: ZeroSizedModule + BuildModule<D, E>, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for T {
    type Built = T;
}

impl<E: Dtype, D: Device<E>, T: ZeroSizedModule> TensorCollection<E, D> for T {
    type To<E2: Dtype, D2: Device<E2>> = T;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields((), |_| Default::default())
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
    type Error = <Self as Module<T>>::Error;

    fn try_forward_mut(&mut self, input: T) -> Result<Self::Output, Self::Error> {
        self.try_forward(input)
    }
}
