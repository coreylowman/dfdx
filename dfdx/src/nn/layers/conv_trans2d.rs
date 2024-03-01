use crate::prelude::*;

/// **Requires Nightly** Performs *unbiased* 2d deconvolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.ConvTranspose2d(..., bias=False)`
///
/// To create a biased conv, combine with [crate::nn::Bias2D].
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
/// - `OutputPadding`: Controls the additional size added to one side of the output shape. Defaults to `Const<0>`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConvTrans2DConfig<
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
    Groups: Dim = Const<1>,
    OutputPadding: Dim = Const<0>,
> {
    pub in_chan: InChan,
    pub out_chan: OutChan,
    pub kernel_size: KernelSize,
    pub stride: Stride,
    pub padding: Padding,
    pub dilation: Dilation,
    pub groups: Groups,
    pub output_padding: OutputPadding,
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
    const OUTPUT_PADDING: usize = 0,
> = ConvTrans2DConfig<
    Const<IN_CHAN>,
    Const<OUT_CHAN>,
    Const<KERNEL_SIZE>,
    Const<STRIDE>,
    Const<PADDING>,
    Const<DILATION>,
    Const<GROUPS>,
    Const<OUTPUT_PADDING>,
>;

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, OP: Dim, E: Dtype, D: Device<E>>
    BuildOnDevice<E, D> for ConvTrans2DConfig<I, O, K, S, P, L, G, OP>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
{
    type Built = ConvTrans2D<I, O, K, S, P, L, G, OP, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        assert_eq!(self.in_chan.size() % self.groups.size(), 0);
        assert_eq!(self.out_chan.size() % self.groups.size(), 0);
        assert!(self.output_padding.size() < self.stride.size());
        let o_over_g = self.out_chan / self.groups;
        let weight =
            device.try_zeros_like(&(self.in_chan, o_over_g, self.kernel_size, self.kernel_size))?;
        Ok(ConvTrans2D {
            weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
            output_padding: self.output_padding,
        })
    }
}

/// See [ConvTrans2DConfig].
#[derive(Debug, Clone, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct ConvTrans2D<
    InChan,
    OutChan,
    KernelSize,
    Stride,
    Padding,
    Dilation,
    Groups,
    OutputPadding,
    Elem,
    Dev,
> where
    OutChan: std::ops::Div<Groups>,
    <OutChan as std::ops::Div<Groups>>::Output: Dim,
    InChan: Dim,
    OutChan: Dim,
    KernelSize: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    OutputPadding: Dim,
    Elem: Dtype,
    Dev: Device<Elem>,
{
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    #[allow(clippy::type_complexity)]
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
    pub output_padding: OutputPadding,
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, OP: Dim, E, D> ResetParams<E, D>
    for ConvTrans2D<I, O, K, S, P, L, G, OP, E, D>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        let (_, o_over_g, k, _) = self.weight.shape();
        let b = (1.0 / (k.size() * k.size() * o_over_g.size()) as f64).sqrt();
        let b = E::from_f64(b).unwrap();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, OP: Dim, E, D, Img> Module<Img>
    for ConvTrans2D<I, O, K, S, P, L, G, OP, E, D>
where
    O: std::ops::Div<G>,
    <O as std::ops::Div<G>>::Output: Dim,
    E: Dtype,
    D: Device<E>,
    (
        Img,
        Tensor<(I, <O as std::ops::Div<G>>::Output, K, K), E, D>,
    ): TryConvTrans2D<S, P, L, G, OP>,
{
    type Output = <(
        Img,
        Tensor<(I, <O as std::ops::Div<G>>::Output, K, K), E, D>,
    ) as TryConvTrans2D<S, P, L, G, OP>>::Convolved;
    fn try_forward(&self, x: Img) -> Result<Self::Output, Error> {
        (x, self.weight.clone()).try_convtrans2d(
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.output_padding,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[rustfmt::skip]
    #[test]
    fn test_forward_3d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank3<3, 8, 8>>();
        let _: Tensor<Rank3<2, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 9, 9>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 11, 11>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 17, 17>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 24, 24>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 6, 6>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 13, 13>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank4<5, 3, 8, 8>>();
        let _: Tensor<Rank4<5, 2, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 9, 9>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 11, 11>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 17, 17>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 24, 24>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 6, 6>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 13, 13>, _, _, _> = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = ConvTrans2DConstConfig<4, 2, 3>;
        type B = ConvTrans2DConstConfig<2, 1, 3>;
        type Model = (A, B);
        let _: Tensor<Rank3<1, 10, 10>, _, _> = dev
            .build_module::<TestDtype>(Model::default())
            .forward(dev.zeros::<Rank3<4, 6, 6>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = ConvTrans2DConstConfig<2, 1, 3>;
        type B = ConvTrans2DConstConfig<4, 2, 3>;
        type C = ConvTrans2DConstConfig<1, 4, 1, 1, 1>;
        type Model = (C, B, A);

        let dev = Cpu::default();
        let _: Tensor<Rank3<1, 10, 10>, _, _> = dev
            .build_module::<TestDtype>(Model::default())
            .forward_mut(dev.zeros::<Rank3<1, 8, 8>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<TestDtype>(<ConvTrans2DConstConfig<2, 4, 3>>::default());

        let weight_init = m.weight.clone();

        let mut opt = crate::nn::optim::Sgd::new(&m, Default::default());
        let out = m.forward(dev.sample_normal::<Rank4<8, 2, 28, 28>>().leaky_trace());
        let g = out.square().mean().backward();

        assert_ne!(
            g.get(&m.weight).array(),
            [[[[TestDtype::zero(); 3]; 3]; 4]; 2]
        );

        opt.update(&mut m, &g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_output_padding() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[[[0.1, 0.7], [0.3, 0.4]]]]);
        let w = dev.tensor([[[[-0.1, -0.3, 0.7], [0.8, -0.2, 0.1], [0.3, 0.4, -0.5]]]]);
        let mut m = dev
            .build_module::<TestDtype>(<ConvTrans2DConstConfig<1, 1, 3, 2, 1, 1, 1, 0>>::default());
        m.weight = w.clone();
        let y: Tensor<Rank4<1, 1, 3, 3>, _, _, _> = m.forward(x.clone());
        assert_close_to_literal!(y,[[[[-0.02, 0.57, -0.14], [-0.05, 0.33, 0.16,], [-0.06, 0.35000002, -0.08000001]]]]);

        let mut m = dev
            .build_module::<TestDtype>(<ConvTrans2DConstConfig<1, 1, 3, 2, 1, 1, 1, 1>>::default());
        m.weight = w.clone();
        let y: Tensor<Rank4<1, 1, 4, 4>, _, _, _> = m.forward(x.clone());
        assert_close_to_literal!(
            y, [[[
                [-0.0200, 0.5700, -0.1400, 0.0700],
                [-0.0500, 0.3300, 0.1600, -0.0700],
                [-0.0600, 0.3500, -0.0800, 0.0400],
                [0.1200, -0.0300, 0.1600, -0.2000],
            ]]]
        );
    }
}
