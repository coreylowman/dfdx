use crate::prelude::*;

/// Adds a learnable 1d bias to 2d `(Batch, I)` and 3d `(Batch, Seq, I)` inputs.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// const I: usize = 5;
/// type Model = Bias1DConstConfig<I>;
/// let model = dev.build_module::<f32>(Model::default());
///
/// // 2d input
/// let x: Tensor<Rank2<5, I>, f32, _> = dev.sample_normal();
/// model.forward(x);
///
/// // 3d input
/// let x: Tensor<Rank3<10, 5, I>, f32, _> = dev.sample_normal();
/// model.forward(x);
/// ```
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bias1DConfig<I: Dim>(pub I);

/// Compile time sugar alias around [Bias1DConfig]
pub type Bias1DConstConfig<const I: usize> = Bias1DConfig<Const<I>>;

impl<I: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Bias1DConfig<I> {
    type Built = Bias1D<I, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(Bias1D {
            bias: device.try_zeros_like(&(self.0,))?,
        })
    }
}

/// See [Bias1DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Bias1D<I: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub bias: Tensor<(I,), Elem, Dev>,
}

impl<I: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for Bias1D<I, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.bias.try_fill_with_zeros()
    }
}

impl<I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(I,), E, D, T>>
    for Bias1D<I, E, D>
{
    type Output = Tensor<(I,), E, D, T>;
    fn try_forward(&self, x: Tensor<(I,), E, D, T>) -> Result<Self::Output, Error> {
        x.try_add(self.bias.clone())
    }
}

impl<Batch: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(Batch, I), E, D, T>>
    for Bias1D<I, E, D>
{
    type Output = Tensor<(Batch, I), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, I), E, D, T>) -> Result<Self::Output, Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, I: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq, I), E, D, T>> for Bias1D<I, E, D>
{
    type Output = Tensor<(Batch, Seq, I), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, Seq, I), E, D, T>) -> Result<Self::Output, Error> {
        self.bias.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}
