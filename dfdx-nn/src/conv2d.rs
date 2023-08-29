use dfdx::{
    shapes::{Const, Dim, Dtype, HasShape},
    tensor::Tensor,
    tensor_ops::{Device, TryConv2D},
};

use crate::*;

/// **Requires Nightly** Performs *unbiased* 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d(..., bias=False)`
///
/// Example usage:
/// ```rust
/// # use dfdx_nn::Conv2DConfig;
/// # use dfdx::shapes::Const;
/// // compile time channels/kernel
/// let m: Conv2DConfig<Const<3>, Const<5>, Const<3>> = Default::default();
/// // runtime channels/kernel
/// let m: Conv2DConfig<usize, usize, usize> = Conv2DConfig {
///     in_chan: 3,
///     out_chan: 5,
///     kernel_size: 3,
///     ..Default::default()
/// };
/// ```
///
/// To create a biased conv, combine with [crate::Bias2D].
///
/// Generics:
/// - `InChan`: The number of input channels in an image.
/// - `OutChan`: The number of channels in the output of the layer.
/// - `KernelSize`: The size of the kernel applied to both width and height of the images.
/// - `Stride`: How far to move the kernel each step. Defaults to `Const<1>`
/// - `Padding`: How much zero padding to add around the images. Defaults to `Const<0>`.
/// - `Dilation`: Controls the spacing between kernel points. Defaults to `Const<1>`.
/// - `Groups`: Controls the connections between inputs and outputs.
///     `InChan` and `OutChan` must both be divisible by `Groups`.
///
/// See [conv animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) for helpful
/// visualization of all of these parameters.
#[derive(Debug, Default, Clone, Copy)]
pub struct Conv2DConfig<
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
    Groups: Dim = Const<1>,
> {
    pub in_chan: InChan,
    pub out_chan: OutChan,
    pub kernel_size: KernelSize,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
    pub groups: Groups,
}

/// Compile time sugar alias around [Conv2DConfig]
pub type Conv2DConstConfig<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
    const GROUPS: usize = 1,
> = Conv2DConfig<
    Const<IN_CHAN>,
    Const<OUT_CHAN>,
    Const<KERNEL_SIZE>,
    Const<STRIDE>,
    Const<PADDING>,
    Const<DILATION>,
    Const<GROUPS>,
>;

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E: Dtype, D: Device<E>>
    crate::BuildOnDevice<E, D> for Conv2DConfig<I, O, K, S, P, L, G>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
{
    type Built = Conv2D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        assert_eq!(self.in_chan.size() % self.groups.size(), 0);
        assert_eq!(self.out_chan.size() % self.groups.size(), 0);
        let i_over_g = self.in_chan / self.groups;
        let weight = device.try_zeros_like(&(
            self.out_chan,
            i_over_g,
            self.kernel_size,
            self.kernel_size,
        ))?;
        Ok(Conv2D {
            weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        })
    }
}

/// The module built with [Conv2DConfig]. See [Conv2DConfig] for usage.
#[derive(Debug, Clone, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct Conv2D<InChan, OutChan, KernelSize, Stride, Padding, Dilation, Groups, Elem, Dev>
where
    InChan: std::ops::Div<Groups>,
    <InChan as std::ops::Div<Groups>>::Output: Dim,
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    Elem: Dtype,
    Dev: Device<Elem>,
{
    #[param]
    #[serialize]
    pub weight: Tensor<
        (
            OutChan,
            <InChan as std::ops::Div<Groups>>::Output,
            KernelSize,
            KernelSize,
        ),
        Elem,
        Dev,
    >,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
    pub groups: Groups,
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D> crate::ResetParams<E, D>
    for Conv2D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let (_, i_over_g, k, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (k.size() * k.size() * i_over_g.size()) as f64).unwrap();
        let b = scale.sqrt();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D, Img> crate::Module<Img>
    for Conv2D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype,
    D: Device<E>,
    (
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ): TryConv2D<S, P, L, G>,
{
    type Output = <(
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConv2D<S, P, L, G>>::Convolved;
    type Error = <(
        Img,
        Tensor<(O, <I as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConv2D<S, P, L, G>>::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        (x, self.weight.clone()).try_conv2d(self.stride, self.padding, self.dilation, self.groups)
    }
}
