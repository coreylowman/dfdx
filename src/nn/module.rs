use crate::prelude::CanUpdateWithGradients;

/// A unit of a neural network. Acts on the generic `Input`
/// and produces `Module::Output`.
///
/// Generic `Input` means you can implement module for multiple
/// input types on the same struct. For example [super::Linear] implements
/// [Module] for 1d inputs and 2d inputs.
pub trait Module<Input>: ResetParams + CanUpdateWithGradients {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Pass an `Input` through the unit and produce [Self::Output].
    /// Can be implemented for multiple `Input` types.
    ///
    /// # Example Usage
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward(Tensor2D::zeros());
    /// ```
    ///
    /// # Example Implementation
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
    /// struct MyMulLayer {
    ///     scale: Tensor1D<5, NoneTape>,
    /// }
    /// # impl Default for MyMulLayer { fn default() -> Self { Self { scale: Tensor1D::zeros() }}}
    /// # impl CanUpdateWithGradients for MyMulLayer { fn update<G: GradientProvider>(&mut self, grads: &mut G, _: &mut UnchangedTensors) { } }
    /// # impl ResetParams for MyMulLayer { fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {}}
    ///
    /// impl Module<Tensor1D<5>> for MyMulLayer {
    ///     type Output = Tensor1D<5>;
    ///     fn forward(&self, input: Tensor1D<5>) -> Self::Output {
    ///         mul(input, &self.scale)
    ///     }
    /// }
    /// ```
    fn forward(&self, input: Input) -> Self::Output;
}

/// Something that can reset it's parameters.
pub trait ResetParams {
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
    ///     fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
    ///         for i in 0..5 {
    ///             self.scale.mut_data()[i] = rng.gen_range(0.0..1.0);
    ///         }
    ///     }
    /// }
    /// ```
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R);
}
