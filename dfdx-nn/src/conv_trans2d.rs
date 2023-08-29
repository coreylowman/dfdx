use dfdx::{
    shapes::{Const, Dim, Dtype, HasShape},
    tensor::Tensor,
    tensor_ops::{Device, TryConvTrans2D},
};

use crate::*;

/// **Requires Nightly** Performs *unbiased* 2d deconvolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.ConvTranspose2d(..., bias=False)`
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
/// - `Groups`: Controls the connections between inputs and outputs. Defaults to `Const<1>`.
///     `InChan` and `OutChan` must both be divisible by `Groups`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConvTrans2DConfig<
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

/// Compile time sugar alias around [ConvTrans2DConfig].
pub type ConvTrans2DConstConfig<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
    const GROUPS: usize = 1,
> = ConvTrans2DConfig<
    Const<IN_CHAN>,
    Const<OUT_CHAN>,
    Const<KERNEL_SIZE>,
    Const<STRIDE>,
    Const<PADDING>,
    Const<DILATION>,
    Const<GROUPS>,
>;

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E: Dtype, D: Device<E>>
    crate::BuildOnDevice<E, D> for ConvTrans2DConfig<I, O, K, S, P, L, G>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
{
    type Built = ConvTrans2D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        assert_eq!(self.in_chan.size() % self.groups.size(), 0);
        assert_eq!(self.out_chan.size() % self.groups.size(), 0);
        let o_over_g = self.out_chan / self.groups;
        let weight =
            device.try_zeros_like(&(self.in_chan, o_over_g, self.kernel_size, self.kernel_size))?;
        Ok(ConvTrans2D {
            weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        })
    }
}

/// See [ConvTrans2DConfig].
#[derive(Debug, Clone, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct ConvTrans2D<InChan, OutChan, KernelSize, Stride, Padding, Dilation, Groups, Elem, Dev>
where
    OutChan: std::ops::Div<Groups>,
    <OutChan as std::ops::Div<Groups>>::Output: Dim,
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
            InChan,
            <OutChan as std::ops::Div<Groups>>::Output,
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
    for ConvTrans2D<I, O, K, S, P, L, G, E, D>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let (_, o_over_g, k, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (k.size() * k.size() * o_over_g.size()) as f64).unwrap();
        let b = scale.sqrt();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D, Img> crate::Module<Img>
    for ConvTrans2D<I, O, K, S, P, L, G, E, D>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
    E: Dtype,
    D: Device<E>,
    (
        Img,
        Tensor<(I, <O as std::ops::Div<G>>::Output, K, K), E, D>,
    ): TryConvTrans2D<S, P, L, G>,
{
    type Output = <(
        Img,
        Tensor<(I, <O as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConvTrans2D<S, P, L, G>>::Convolved;
    type Error = <(
        Img,
        Tensor<(I, <O as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConvTrans2D<S, P, L, G>>::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        (x, self.weight.clone()).try_convtrans2d(
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}
