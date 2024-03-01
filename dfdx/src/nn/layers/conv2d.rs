use crate::prelude::*;

/// **Requires Nightly** Performs *unbiased* 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d(..., bias=False)`
///
/// Example usage:
/// ```rust
/// # use dfdx::nn::Conv2DConfig;
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
/// To create a biased conv, combine with [crate::nn::Bias2D].
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
    BuildOnDevice<E, D> for Conv2DConfig<I, O, K, S, P, L, G>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
{
    type Built = Conv2D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
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
#[derive(Debug, Clone, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
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
    #[cfg_attr(feature = "safetensors", serialize)]
    #[allow(clippy::type_complexity)]
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

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D> ResetParams<E, D>
    for Conv2D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        let (_, i_over_g, k, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (k.size() * k.size() * i_over_g.size()) as f64).unwrap();
        let b = scale.sqrt();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D, Img> Module<Img>
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
    fn try_forward(&self, x: Img) -> Result<Self::Output, Error> {
        (x, self.weight.clone()).try_conv2d(self.stride, self.padding, self.dilation, self.groups)
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
        let x = dev.zeros::<Rank3<3, 10, 10>>();
        let _: Tensor<Rank3<2, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 9, 9>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<4, 7, 7>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 4, 4>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 3, 3>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 12, 12>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<2, 6, 6>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[test]
    fn test_grouped_forward_sizes() {
        let dev: TestDevice = Default::default();

        let x = dev.zeros::<Rank3<16, 10, 10>>();

        let m =
            dev.build_module::<TestDtype>(<Conv2DConstConfig<16, 32, 3, 1, 0, 1, 1>>::default());
        let _: Tensor<Rank4<32, 16, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv2DConstConfig<16, 32, 3, 1, 0, 1, 2>>::default());
        let _: Tensor<Rank4<32, 8, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv2DConstConfig<16, 32, 3, 1, 0, 1, 4>>::default());
        let _: Tensor<Rank4<32, 4, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv2DConstConfig<16, 32, 3, 1, 0, 1, 8>>::default());
        let _: Tensor<Rank4<32, 2, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv2DConstConfig<16, 32, 3, 1, 0, 1, 16>>::default());
        let _: Tensor<Rank4<32, 1, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x);
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank4<5, 3, 10, 10>>();
        let _: Tensor<Rank4<5, 2, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 8, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 9, 9>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 7, 7>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 4, 4>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 3, 3>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 10, 10>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 12, 12>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 6, 6>, _, _, _> = dev.build_module::<TestDtype>(<Conv2DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = Conv2DConstConfig<1, 2, 3>;
        type B = Conv2DConstConfig<2, 4, 3>;
        let _: Tensor<Rank3<4, 6, 6>, _, _> = dev
            .build_module::<TestDtype>(<(A, B)>::default())
            .forward(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2DConstConfig<1, 2, 3>;
        type B = Conv2DConstConfig<2, 4, 3>;
        type C = Conv2DConstConfig<4, 1, 1, 1, 1>;

        let dev = Cpu::default();
        let _: Tensor<Rank3<1, 8, 8>, _, _> = dev
            .build_module::<TestDtype>(<(A, B, C)>::default())
            .forward_mut(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<TestDtype>(Conv2DConstConfig::<2, 4, 3>::default());

        let weight_init = m.weight.clone();

        let mut opt = crate::nn::optim::Sgd::new(&m, Default::default());
        let out = m.forward(dev.sample_normal::<Rank4<8, 2, 28, 28>>().leaky_trace());
        let g = out.square().mean().backward();

        assert_ne!(
            g.get(&m.weight).array(),
            [[[[TestDtype::zero(); 3]; 3]; 2]; 4]
        );

        opt.update(&mut m, &g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
    }
}
