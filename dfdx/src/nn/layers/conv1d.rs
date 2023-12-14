use crate::prelude::*;

/// **Requires Nightly** Performs *unbiased* 1d convolutions on 2d and 3d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv1d(..., bias=False)`
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
/// - `DILATION`: Controls the spacing between kernel points. Defaults to `1`.
/// - `GROUPS`: Controls the connections between inputs and outputs.
///     `IN_CHAN` and `OUT_CHAN` must both be divisible by `GROUPS`. For example,
///
/// See [conv animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) for helpful
/// visualization of all of these parameters.

#[derive(Debug, Default, Clone, Copy)]
pub struct Conv1DConfig<
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

/// Compile time sugar alias around [Conv1DConfig]
pub type Conv1DConstConfig<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
    const GROUPS: usize = 1,
> = Conv1DConfig<
    Const<IN_CHAN>,
    Const<OUT_CHAN>,
    Const<KERNEL_SIZE>,
    Const<STRIDE>,
    Const<PADDING>,
    Const<DILATION>,
    Const<GROUPS>,
>;

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E: Dtype, D: Device<E>>
    BuildOnDevice<E, D> for Conv1DConfig<I, O, K, S, P, L, G>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
{
    type Built = Conv1D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        assert_eq!(self.in_chan.size() % self.groups.size(), 0);
        assert_eq!(self.out_chan.size() % self.groups.size(), 0);
        let i_over_g = self.in_chan / self.groups;
        let weight = device.try_zeros_like(&(self.out_chan, i_over_g, self.kernel_size))?;
        Ok(Conv1D {
            weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        })
    }
}

/// The module built with [Conv1DConfig]. See [Conv1DConfig] for usage.
#[derive(Debug, Clone, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Conv1D<InChan, OutChan, KernelSize, Stride, Padding, Dilation, Groups, Elem, Dev>
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
    for Conv1D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        let (_, i_over_g, k) = self.weight.shape();
        let scale = (1.0 / (k.size() * i_over_g.size()) as f64).sqrt();
        let b = E::from_f64(scale).unwrap();
        self.weight
            .try_fill_with_distr(rand_distr::Uniform::new(-b, b))
    }
}

impl<I: Dim, O: Dim, K: Dim, S: Dim, P: Dim, L: Dim, G: Dim, E, D, Img> Module<Img>
    for Conv1D<I, O, K, S, P, L, G, E, D>
where
    I: std::ops::Div<G>,
    <I as std::ops::Div<G>>::Output: Dim,
    E: Dtype,
    D: Device<E>,
    (Img, Tensor<(O, <I as std::ops::Div<G>>::Output, K), E, D>): TryConv1D<S, P, L, G>,
{
    type Output = <(Img, Tensor<(O, <I as std::ops::Div<G>>::Output, K), E, D>) as TryConv1D<
        S,
        P,
        L,
        G,
    >>::Convolved;
    fn try_forward(&self, x: Img) -> Result<Self::Output, Error> {
        (x, self.weight.clone()).try_conv1d(self.stride, self.padding, self.dilation, self.groups)
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
        let x = dev.zeros::<Rank2<3, 10>>();
        let _: Tensor<Rank2<2, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank2<4, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank2<4, 9>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank2<4, 7>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank2<2, 4>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank2<2, 3>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank2<2, 10>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank2<2, 12>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank2<2, 6>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[test]
    fn test_grouped_forward_sizes() {
        let dev: TestDevice = Default::default();

        let x = dev.ones::<Rank2<16, 10>>();

        let m = dev.build_module::<TestDtype>(<Conv1DConstConfig<16, 32, 3, 1, 0, 1>>::default());
        let _: Tensor<Rank3<32, 16, 3>, _, _> = m.weight;
        let _: Tensor<Rank2<32, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv1DConstConfig<16, 32, 3, 1, 0, 1, 2>>::default());
        let _: Tensor<Rank3<32, 8, 3>, _, _> = m.weight;
        let _: Tensor<Rank2<32, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv1DConstConfig<16, 32, 3, 1, 0, 1, 4>>::default());
        let _: Tensor<Rank3<32, 4, 3>, _, _> = m.weight;
        let _: Tensor<Rank2<32, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv1DConstConfig<16, 32, 3, 1, 0, 1, 8>>::default());
        let _: Tensor<Rank3<32, 2, 3>, _, _> = m.weight;
        let _: Tensor<Rank2<32, 8>, _, _> = m.forward(x.clone());

        let m =
            dev.build_module::<TestDtype>(<Conv1DConstConfig<16, 32, 3, 1, 0, 1, 16>>::default());
        let _: Tensor<Rank3<32, 1, 3>, _, _> = m.weight;
        let _: Tensor<Rank2<32, 8>, _, _> = m.forward(x);
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank3<5, 3, 10>>();
        let _: Tensor<Rank3<5, 2, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 4, 8>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 4, 9>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 4, 7>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 4, 4>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 2, 4>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 2, 3>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 3>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 2, 10>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 1, 1>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 2, 12>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 1, 2>>::default()).forward(x.clone());
        let _: Tensor<Rank3<5, 2, 6>, _, _, _> = dev.build_module::<TestDtype>(<Conv1DConstConfig<3, 2, 3, 2, 2>>::default()).forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = Conv1DConstConfig<1, 2, 3>;
        type B = Conv1DConstConfig<2, 4, 3>;
        let _: Tensor<Rank2<4, 6>, _, _> = dev
            .build_module::<TestDtype>(<(A, B)>::default())
            .forward(dev.zeros::<Rank2<1, 10>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv1DConstConfig<1, 2, 3>;
        type B = Conv1DConstConfig<2, 4, 3>;
        type C = Conv1DConstConfig<4, 1, 1, 1, 1>;

        let dev = Cpu::default();
        let _: Tensor<Rank2<1, 8>, _, _> = dev
            .build_module::<TestDtype>(<(A, B, C)>::default())
            .forward_mut(dev.zeros::<Rank2<1, 10>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<TestDtype>(<Conv1DConstConfig<2, 4, 3>>::default());

        let weight_init = m.weight.clone();

        let mut opt = crate::nn::optim::Sgd::new(&m, Default::default());
        let out = m.forward(dev.sample_normal::<Rank3<8, 2, 28>>().leaky_trace());
        let g = out.square().mean().backward();

        assert_ne!(g.get(&m.weight).array(), [[[TestDtype::zero(); 3]; 2]; 4]);

        opt.update(&mut m, &g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
    }
}
