use rand_distr::{uniform::SampleUniform, Uniform};
use std::string::String;

use crate::{
    prelude::Tensor,
    shapes::{Dtype, Shape},
    tensor_ops::Device,
};

#[cfg(feature = "cuda")]
pub use crate::tensor::OnCuda;
pub use crate::tensor::{DeviceStorage, OnCpu, OnDevice, ToDevice};

use super::{
    visit_tensors::VisitTensors, TensorFunction, TensorFunctionOption, TensorVisitor,
    VisitTensorGroups, VisitTensorsMut,
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

struct ResetParamsFunction {}

impl<E: Dtype + SampleUniform, D: Device<E>> TensorFunction<0, 1, E, D> for ResetParamsFunction {
    type Err = D::Err;

    fn call<S: Shape>(
        &mut self,
        _refs: [&Tensor<S, E, D>; 0],
        refs_mut: [&mut Tensor<S, E, D>; 1],
        _name: Option<std::string::String>,
        options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err> {
        for o in options.iter().rev() {
            match o {
                TensorFunctionOption::ResetParamsUniform(a, b) => {
                    let distr = Uniform::new(E::from_f64(*a).unwrap(), E::from_f64(*b).unwrap());
                    return refs_mut[0].try_fill_with_distr(distr);
                }
                TensorFunctionOption::ResetParamsOnes => {
                    return refs_mut[0].try_fill_with_ones();
                }
                _ => {}
            }
        }

        refs_mut[0].try_fill_with_zeros()
    }
}

/// Something that can reset it's parameters.
pub trait ResetParams<D: Device<E>, E: Dtype + SampleUniform>: VisitTensorsMut<E, D> {
    /// Mutates parameters. Each implementor
    /// of this trait decides how the parameters are initialized. In
    /// fact, some impls may not even use randomness.
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }

    /// Fallible version of [ResetParams::reset_params].
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.visit_mut(&mut ResetParamsFunction {})
    }
}

impl<D: Device<E>, E: Dtype + SampleUniform, T: VisitTensorsMut<E, D>> ResetParams<D, E> for T {}

struct CountParamsFunction(usize);

impl<E: Dtype, D: DeviceStorage> TensorFunction<1, 0, E, D> for CountParamsFunction {
    type Err = String;

    fn call<S: Shape>(
        &mut self,
        refs: [&Tensor<S, E, D>; 1],
        _refs_mut: [&mut Tensor<S, E, D>; 0],
        _name: Option<String>,
        _options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err> {
        self.0 += refs[0].shape().num_elements();
        Ok(())
    }
}

/// Computes the number of parameters present in a module
pub trait CountParams<E: Dtype, D: DeviceStorage>: VisitTensors<E, D> {
    /// Returns the total size of all tensors in self. This does not distinguish trainable
    /// parameters from non-trainable parameters.
    ///
    /// Example:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let model = dev.build_module::<Linear<5, 2>, f32>();
    /// assert_eq!(model.param_count(), 12); // 10 in weight, 2 in bias
    /// ```
    fn param_count(&self) -> usize {
        let mut visitor = CountParamsFunction(0);
        // CountParamsVisitor::call never returns Err
        self.visit(&mut visitor).unwrap();
        visitor.0
    }
}

impl<E: Dtype, D: DeviceStorage, T: VisitTensors<E, D>> CountParams<E, D> for T {}

/// Marker trait for modules with no updatable parameters. These have
/// blanket impls for [ResetParams], [GradientUpdate], and [ModuleMut]
pub trait ZeroSizedModule: Default {}

impl<const N: usize, const M: usize, E: Dtype, D: DeviceStorage, T: ZeroSizedModule>
    VisitTensorGroups<N, M, E, D> for T
{
    fn visit_groups<F: TensorFunction<N, M, E, D>>(
        _visitor: TensorVisitor<N, M, Self, F>,
    ) -> Result<(), F::Err> {
        Ok(())
    }
}

impl<T: ZeroSizedModule + BuildModule<D, E>, D: DeviceStorage, E: Dtype> BuildOnDevice<D, E> for T {
    type Built = T;
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
