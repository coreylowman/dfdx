/// A unit of a neural network that accepts `&self`.
/// Acts on the generic `Input` and produces [Module::Output].
///
/// For specifying different behavior based on ownership, see [ModuleMut].
pub trait Module<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Pass an `Input` through the unit and produce [Self::Output].
    /// Can be implemented for multiple `Input` types.
    ///
    /// This should never change `self`.
    /// **See [ModuleMut] for version that can mutate `self`.**
    ///
    /// Example Usage:
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let model: Linear<7, 2> = Default::default();
    /// let y1: Tensor1D<2> = model.forward(Tensor1D::zeros());
    /// let y2: Tensor2D<10, 2> = model.forward(Tensor2D::zeros());
    /// ```
    ///
    /// Example Implementation:
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
    /// struct MyMulLayer {
    ///     scale: Tensor1D<5>,
    /// }
    /// # impl Default for MyMulLayer { fn default() -> Self { Self { scale: Tensor1D::zeros() }}}
    /// # impl CanUpdateWithGradients for MyMulLayer { fn update<G: GradientProvider>(&mut self, grads: &mut G, _: &mut UnusedTensors) { } }
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

/// A unit of a neural network that accepts `&mut self`.
/// Acts on the generic `Input` and produces [ModuleMut::Output].
///
/// For specifying different behavior based on ownership, see [Module].
pub trait ModuleMut<Input> {
    /// The type that this unit produces given `Input`.
    type Output;

    /// Pass an `Input` through the unit and produce [Self::Output].
    /// Can be implemented for multiple `Input` types.
    ///
    /// This *can* change `self`. **See [Module::forward()] for immutable version**
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
