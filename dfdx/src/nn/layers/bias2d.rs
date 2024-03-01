use crate::prelude::*;

/// Adds a learnable 1d bias to 3d `(C, Height, Width)` and 4d `(Batch, C, Height, Width)` inputs.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// const NUM_CHANS: usize = 5;
/// type Model = Bias2DConstConfig<NUM_CHANS>;
/// let model = dev.build_module::<f32>(Model::default());
///
/// // 3d input
/// let x: Tensor<Rank3<NUM_CHANS, 2, 3>, f32, _> = dev.sample_normal();
/// model.forward(x);
///
/// // 4d input
/// let x: Tensor<Rank4<10, NUM_CHANS, 2, 3>, f32, _> = dev.sample_normal();
/// model.forward(x);
/// ```
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bias2DConfig<C: Dim>(pub C);

/// Compile time sugar alias around [Bias2DConfig]
pub type Bias2DConstConfig<const C: usize> = Bias2DConfig<Const<C>>;

impl<C: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Bias2DConfig<C> {
    type Built = Bias2D<C, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(Bias2D {
            bias: device.try_zeros_like(&(self.0,))?,
        })
    }
}

/// See [Bias2DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Bias2D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub bias: Tensor<(C,), Elem, Dev>,
}

impl<C: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for Bias2D<C, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.bias.try_fill_with_zeros()
    }
}

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for Bias2D<C, E, D>
{
    type Output = Tensor<(C, H, W), E, D, T>;
    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(B, C, H, W), E, D, T>> for Bias2D<C, E, D>
{
    type Output = Tensor<(B, C, H, W), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
