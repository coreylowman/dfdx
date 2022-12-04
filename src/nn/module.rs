use crate::{arrays::Dtype, optim::CanUpdateWithGradients, tensor_ops::Device};

/// Immutable forward of `Input` that produces [Module::Output].
/// See [ModuleMut] for mutable forward.
pub trait Module<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Forward `Input` through the module and produce [Module::Output].
    ///
    /// **See [ModuleMut] for version that can mutate `self`.**
    ///
    /// Example Usage:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward(Tensor2D::zeros());
    /// ```
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
    ///
    /// Example Usage:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let mut model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward_mut(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward_mut(Tensor2D::zeros());
    /// ```
    fn forward_mut(&mut self, input: Input) -> Self::Output;
}

/// Something that can reset it's parameters.
pub trait BuildModule<D: Device<E>, E: Dtype>: Sized {
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }

    fn try_build(device: &D) -> Result<Self, D::Err>;

    /// Mutate the unit's parameters using [rand::Rng]. Each implementor
    /// of this trait decides how the parameters are initialized. In
    /// fact, some impls may not even use the `rng`.
    ///
    /// # Example:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// struct MyMulLayer {
    ///     scale: Tensor1D<5, NoneTape>,
    /// }
    ///
    /// impl ResetParams for MyMulLayer {
    ///     fn reset_params(&mut self) {
    ///         for i in 0..5 {
    ///             self.scale.mut_data()[i] = rng.gen_range(0.0..1.0);
    ///         }
    ///     }
    /// }
    /// ```
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }

    fn try_reset_params(&mut self) -> Result<(), D::Err>;
}

pub trait ZeroSizedModule: Default {}

impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> BuildModule<D, E> for T {
    fn try_build(_: &D) -> Result<Self, <D>::Err> {
        Ok(Default::default())
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        Ok(())
    }
}

impl<T: ZeroSizedModule, D: Device<E>, E: Dtype> CanUpdateWithGradients<D, E> for T {
    fn update<U>(&mut self, _: &mut U, _: &mut crate::optim::UnusedTensors) -> Result<(), <D>::Err>
    where
        U: crate::optim::UpdateParams<D, E>,
    {
        Ok(())
    }
}
