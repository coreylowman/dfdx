use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{gradients::Tape, shapes::*, tensor::visitors::*, tensor::*, tensor_ops::*};

use super::{BuildModule, BuildOnDevice, Module, ModuleMut, ToDevice};

pub mod builder {
    #[derive(Debug)]
    pub struct Conv2D<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize = 1,
        const PADDING: usize = 0,
    >;
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D>
    BuildOnDevice<D, E> for builder::Conv2D<I, O, K, S, P>
where
    E: Dtype,
    D: Device<E>,
    Conv2D<I, O, K, S, P, E, D>: BuildModule<D, E>,
{
    type Built = Conv2D<I, O, K, S, P, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// **Requires Nightly** Performs 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d`
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
#[derive(Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub weight: Tensor<Rank4<OUT_CHAN, IN_CHAN, KERNEL_SIZE, KERNEL_SIZE>, E, D>,
    pub bias: Tensor<Rank1<OUT_CHAN>, E, D>,
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D>
    TensorCollection<E, D> for Conv2D<I, O, K, S, P, E, D>
where
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
{
    fn iter_tensors<V: ModuleWalker<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            |s| &s.weight,
            |s| &mut s.weight,
            TensorOptions::named("weight", |t| {
                let b = E::ONE / E::from_usize(I * K * K).unwrap().sqrt();
                t.try_fill_with_distr(rand_distr::Uniform::new(-b, b))
            }),
        )?;
        visitor.visit_tensor(
            |s| &s.bias,
            |s| &mut s.bias,
            TensorOptions::named("bias", |t| {
                let b = E::ONE / E::from_usize(I * K * K).unwrap().sqrt();
                t.try_fill_with_distr(rand_distr::Uniform::new(-b, b))
            }),
        )
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D>
    BuildModule<D, E> for Conv2D<I, O, K, S, P, E, D>
where
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        let k = E::from_usize(I * K * K).unwrap();
        let bound = E::ONE / k.sqrt();
        Ok(Self {
            weight: device.try_sample(rand_distr::Uniform::new(-bound, bound))?,
            bias: device.try_sample(rand_distr::Uniform::new(-bound, bound))?,
        })
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D1, D2>
    ToDevice<D2> for Conv2D<I, O, K, S, P, E, D1>
where
    E: Dtype,
    D1: Device<E>,
    D2: Device<E>,
{
    type Output = Conv2D<I, O, K, S, P, E, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        Conv2D {
            weight: self.weight.to_device(device),
            bias: self.bias.to_device(device),
        }
    }
}

#[cfg(feature = "nightly")]
impl<const C: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D, Img>
    Module<Img> for Conv2D<C, O, K, S, P, E, D>
where
    E: Dtype,
    D: Device<E>,
    Img: TryConv2DTo<Tensor<Rank4<O, C, K, K>, E, D>, S, P>,
    for<'a> Bias2D<'a, O, E, D>: Module<Img::Output, Output = Img::Output>,
{
    type Output = Img::Output;
    fn forward(&self, x: Img) -> Self::Output {
        Bias2D { beta: &self.bias }.forward(x.conv2d_to(self.weight.clone()))
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, E, D, Img>
    ModuleMut<Img> for Conv2D<I, O, K, S, P, E, D>
where
    E: Dtype,
    D: Device<E>,
    Self: Module<Img>,
{
    type Output = <Self as Module<Img>>::Output;
    fn forward_mut(&mut self, input: Img) -> Self::Output {
        self.forward(input)
    }
}

#[derive(Clone, Debug)]
struct Bias2D<'a, const C: usize, E: Dtype, D: DeviceStorage> {
    beta: &'a Tensor<Rank1<C>, E, D>,
}

impl<'a, const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(Const<C>, H, W), E, D, T>> for Bias2D<'a, C, E, D>
{
    type Output = Tensor<(Const<C>, H, W), E, D, T>;
    fn forward(&self, input: Tensor<(Const<C>, H, W), E, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

impl<'a, B: Dim, const C: usize, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(B, Const<C>, H, W), E, D, T>> for Bias2D<'a, C, E, D>
{
    type Output = Tensor<(B, Const<C>, H, W), E, D, T>;
    fn forward(&self, input: Tensor<(B, Const<C>, H, W), E, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use crate::{
        nn::DeviceBuildExt,
        optim::*,
        tensor::{AsArray, SampleTensor, ZerosTensor},
        tests::*,
    };

    use super::{builder::Conv2D, *};

    #[rustfmt::skip]
    #[test]
    fn test_forward_3d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank3<3, 10, 10>>();
        let _: Tensor<Rank3<2, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 4, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 9, 9>, _, _, _> = dev.build_module::<Conv2D<3, 4, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 7, 7>, _, _, _> = dev.build_module::<Conv2D<3, 4, 4>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 4, 4>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 3, 3>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 10, 10>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 1>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 12, 12>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 6, 6>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2, 2>, TestDtype>().forward(x.clone());
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank4<5, 3, 10, 10>>();
        let _: Tensor<Rank4<5, 2, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 4, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 9, 9>, _, _, _> = dev.build_module::<Conv2D<3, 4, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 7, 7>, _, _, _> = dev.build_module::<Conv2D<3, 4, 4>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 4, 4>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 3, 3>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 10, 10>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 1>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 12, 12>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 6, 6>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2, 2>, TestDtype>().forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        let _: Tensor<Rank3<4, 6, 6>, _, _> = dev
            .build_module::<(A, B), TestDtype>()
            .forward(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        type C = Conv2D<4, 1, 1, 1, 1>;

        let dev = Cpu::default();
        let _: Tensor<Rank3<1, 8, 8>, _, _> = dev
            .build_module::<(A, B, C), TestDtype>()
            .forward_mut(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<Conv2D<2, 4, 3>, TestDtype>();

        let weight_init = m.weight.clone();
        let bias_init = m.bias.clone();

        let mut opt = Sgd::new(&m, Default::default());
        let out = m.forward(dev.sample_normal::<Rank4<8, 2, 28, 28>>().trace());
        let g = out.square().mean().backward();

        assert_ne!(g.get(&m.weight).array(), [[[[0.0; 3]; 3]; 2]; 4]);
        assert_ne!(g.get(&m.bias).array(), [0.0; 4]);

        opt.update(&mut m, g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
        assert_ne!(bias_init.array(), m.bias.array());
    }
}
