use crate::gradients::{CanUpdateWithGradients, GradientProvider, Tape, UnusedTensors};
use crate::prelude::*;
use rand::Rng;
use rand_distr::Uniform;

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
///
/// Examples:
/// ```rust
/// #![feature(generic_const_exprs)]
/// # use dfdx::prelude::*;
/// let m: Conv2D<16, 33, 3> = Default::default();
/// let _: Tensor3D<33, 30, 62> = m.forward(Tensor3D::<16, 32, 64>::zeros());
/// let _: Tensor4D<2, 33, 13, 12> = m.forward(Tensor4D::<2, 16, 15, 14>::zeros());
/// ```
#[derive(Default, Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
> {
    pub weight: Tensor4D<OUT_CHAN, IN_CHAN, KERNEL_SIZE, KERNEL_SIZE>,
    pub bias: Tensor1D<OUT_CHAN>,
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize>
    CanUpdateWithGradients for Conv2D<I, O, K, S, P>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.weight.update(grads, unused);
        self.bias.update(grads, unused);
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize> ResetParams
    for Conv2D<I, O, K, S, P>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        let k = (I * K * K) as f32;
        let bound = 1.0 / k.sqrt();
        let dist = Uniform::new(-bound, bound);
        self.weight.randomize(rng, &dist);
        self.bias.randomize(rng, &dist);
    }
}

impl<
        T: Tape,
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor3D<I, H, W, T>> for Conv2D<I, O, K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor3D<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor3D<I, H, W, T>) -> Self::Output {
        x.conv2d::<O, K, S, P>(&self.weight, &self.bias)
    }
}

impl<
        T: Tape,
        const B: usize,
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor4D<B, I, H, W, T>> for Conv2D<I, O, K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor4D<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor4D<B, I, H, W, T>) -> Self::Output {
        x.conv2d::<O, K, S, P>(&self.weight, &self.bias)
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, T> ModuleMut<T>
    for Conv2D<I, O, K, S, P>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_forward_3d_sizes() {
        type Img = Tensor3D<3, 10, 10>;
        let _: Tensor3D<2, 8, 8> = Conv2D::<3, 2, 3>::default().forward(Img::zeros());
        let _: Tensor3D<4, 8, 8> = Conv2D::<3, 4, 3>::default().forward(Img::zeros());
        let _: Tensor3D<4, 9, 9> = Conv2D::<3, 4, 2>::default().forward(Img::zeros());
        let _: Tensor3D<4, 7, 7> = Conv2D::<3, 4, 4>::default().forward(Img::zeros());
        let _: Tensor3D<2, 4, 4> = Conv2D::<3, 2, 3, 2>::default().forward(Img::zeros());
        let _: Tensor3D<2, 3, 3> = Conv2D::<3, 2, 3, 3>::default().forward(Img::zeros());
        let _: Tensor3D<2, 10, 10> = Conv2D::<3, 2, 3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor3D<2, 12, 12> = Conv2D::<3, 2, 3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor3D<2, 6, 6> = Conv2D::<3, 2, 3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_forward_4d_sizes() {
        type Img = Tensor4D<5, 3, 10, 10>;
        let _: Tensor4D<5, 2, 8, 8> = Conv2D::<3, 2, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 8, 8> = Conv2D::<3, 4, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 9, 9> = Conv2D::<3, 4, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 4, 7, 7> = Conv2D::<3, 4, 4>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 4, 4> = Conv2D::<3, 2, 3, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 3, 3> = Conv2D::<3, 2, 3, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 10, 10> = Conv2D::<3, 2, 3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 12, 12> = Conv2D::<3, 2, 3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 2, 6, 6> = Conv2D::<3, 2, 3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_2_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        let _: Tensor3D<4, 6, 6> = <(A, B)>::default().forward(Tensor3D::<1, 10, 10>::zeros());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        type C = Conv2D<4, 1, 1, 1, 1>;

        type Img = Tensor3D<1, 10, 10>;
        let _: Tensor3D<1, 8, 8> = <(A, B, C)>::default().forward_mut(Img::zeros());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let mut rng = thread_rng();

        let mut m: Conv2D<2, 4, 3> = Default::default();
        m.reset_params(&mut rng);

        let weight_init = m.weight.clone();
        let bias_init = m.bias.clone();

        let mut opt: Sgd<_> = Default::default();
        let out = m.forward(Tensor4D::<8, 2, 28, 28>::randn(&mut rng).trace());
        let gradients = backward(out.square().mean());

        assert_ne!(gradients.ref_gradient(&m.weight), &[[[[0.0; 3]; 3]; 2]; 4]);
        assert_ne!(gradients.ref_gradient(&m.bias), &[0.0; 4]);

        opt.update(&mut m, gradients).expect("unused params");

        assert_ne!(weight_init.data(), m.weight.data());
        assert_ne!(bias_init.data(), m.bias.data());
    }
}
