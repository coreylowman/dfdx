// use crate::gradients::{CanUpdateWithGradients, GradientProvider, Tape, UnusedTensors};
// use crate::prelude::*;
// use rand::Rng;
// use rand_distr::Uniform;

use crate::{arrays::*, gradients::Tape, optim::*, tensor::*, tensor_ops::*};

use super::module::{Module, ModuleMut, ResetParams};

/// A linear transformation of the form `weight * x + bias`, where `weight` is a matrix, `x` is a vector or matrix,
/// and `bias` is a vector.
///
/// # Generics
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// # Examples
/// `Linear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```rust
/// # use dfdx::prelude::*;
/// let model: Linear<5, 2> = Default::default();
/// assert_eq!(model.weight.data(), &[[0.0; 5]; 2]);
/// assert_eq!(model.bias.data(), &[0.0; 2]);
/// let x: Tensor1D<5> = Default::default();
/// let y: Tensor1D<2> = model.forward(x);
/// assert_eq!(y.data(), &[0.0; 2]);
/// ```
#[derive(Debug, Clone)]
pub struct Linear<const I: usize, const O: usize, D: Device<f32>> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<O, I>, f32, D>,

    /// Bias vector, shape (O, )
    pub bias: Tensor<Rank1<O>, f32, D>,
}

impl<const I: usize, const O: usize, D: Device<f32>> CanUpdateWithGradients<D, f32>
    for Linear<I, O, D>
{
    fn update<P: ParamUpdater<D, f32>>(
        &mut self,
        opt: &mut P,
        unused: &mut UnusedTensors,
    ) -> Result<(), D::Err> {
        self.weight.update(opt, unused)?;
        self.bias.update(opt, unused)?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, D: Device<f32>> ResetParams for Linear<I, O, D> {
    /// Initializes [Self::weight] and [Self::bias] from a [Uniform] distribution
    /// between [-1 / sqrt(I), 1 / sqrt(I)].
    ///
    /// This uses [Randomize::randomize()] to set the values of the tensor.
    fn reset_params(&mut self) {
        todo!();
        //     let bound: f32 = 1.0 / (I as f32).sqrt();
        //     let dist = Uniform::new(-bound, bound);
        //     self.weight.randomize(rng, &dist);
        //     self.bias.randomize(rng, &dist);
    }
}

impl<const I: usize, const O: usize, D: Device<f32>, T: Tape<D>> Module<Tensor<Rank1<I>, f32, D, T>>
    for Linear<I, O, D>
{
    type Output = Tensor<Rank1<O>, f32, D, T>;

    /// 1d forward using [vecmat_mul()] and [add()].
    fn forward(&self, x: Tensor<Rank1<I>, f32, D, T>) -> Self::Output {
        x.matmul(self.weight.retaped::<T>().permute()) + self.bias.clone()
    }
}

impl<const B: usize, const I: usize, const O: usize, D: Device<f32>, T: Tape<D>>
    Module<Tensor<Rank2<B, I>, f32, D, T>> for Linear<I, O, D>
{
    type Output = Tensor<Rank2<B, O>, f32, D, T>;

    /// Batched 2d forward using [matmul()] and [add()]
    fn forward(&self, x: Tensor<Rank2<B, I>, f32, D, T>) -> Self::Output {
        x.matmul(self.weight.retaped::<T>().permute()) + self.bias.retaped::<T>().broadcast()
    }
}

impl<
        const B: usize,
        const S: usize,
        const I: usize,
        const O: usize,
        D: Device<f32>,
        T: Tape<D>,
    > Module<Tensor<Rank3<B, S, I>, f32, D, T>> for Linear<I, O, D>
{
    type Output = Tensor<Rank3<B, S, O>, f32, D, T>;

    /// Batched 3d forward using [matmul()] and [add()]
    fn forward(&self, x: Tensor<Rank3<B, S, I>, f32, D, T>) -> Self::Output {
        x.matmul(self.weight.retaped::<T>().permute()) + self.bias.retaped::<T>().broadcast()
    }
}

impl<T, const I: usize, const O: usize, D: Device<f32>> ModuleMut<T> for Linear<I, O, D>
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
    use crate::tests::{assert_close, build_test_device};

    const W: [[f32; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f32; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_forward_1d() {
        let dev = build_test_device!();

        let model = Linear {
            weight: dev.tensor(W),
            bias: dev.tensor(B),
        };

        let x = dev.tensor([-0.8808001f32, 2.4185333, 2.2478335, 0.0565211, 2.031299]);
        let y = model.forward(x.trace());
        assert_close(&y.as_array(), &[-0.93430865, 0.08624211]);

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).as_array(),
            &[
                [0.82293916, -2.2596567, -2.1001704, -0.05280815, -1.8978603],
                [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
            ],
        );
        assert_close(&g.get(&model.bias).as_array(), &[-0.93430865, 0.08624211]);
    }

    #[test]
    fn test_forward_2d() {
        let dev = build_test_device!();

        let model = Linear {
            weight: dev.tensor(W),
            bias: dev.tensor(B),
        };

        let x = dev.tensor([
            [-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643],
            [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705],
            [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455],
        ]);
        let y = model.forward(x.trace());
        assert_close(
            &y.as_array(),
            &[
                [1.3914378, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872],
            ],
        );

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).as_array(),
            &[
                [-1.1541969, 0.6956873, -0.8553807, 0.9289255, 0.04931633],
                [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.5286576],
            ],
        );
        assert_close(&g.get(&model.bias).as_array(), &[0.7679174, -0.31687993]);
    }

    #[test]
    fn test_forward_3d() {
        let dev = build_test_device!();

        let model = Linear {
            weight: dev.tensor(W),
            bias: dev.tensor(B),
        };

        #[rustfmt::skip]
        let x = dev.tensor([
            [[-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643], [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705], [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455]],
            [[1.2879219, 0.70150787, -1.6746868, 1.7261779, -0.94021803], [-2.6883178, 2.9369607, 2.9256766, 0.27559614, -0.17530347], [0.17499207, -0.11440835, 0.16627812, -0.91773695, 1.1128315]],
        ]);
        let y = model.forward(x.trace());
        assert_close(
            &y.as_array(),
            &[
                [
                    [1.3914378, -0.012851536],
                    [-0.005462587, -0.14800104],
                    [0.9177769, -0.7897872],
                ],
                [
                    [0.6118467, -0.2024638],
                    [-0.6274954, -0.56451213],
                    [0.12783213, -0.0068387985],
                ],
            ],
        );

        let g = y.square().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&model.weight).as_array(),
            &[[-0.16088384, 0.10978711, -0.9008978, 0.59211355, -0.029177088], [0.35563633, -0.38838047, -0.17600831, -0.2034213, 0.31128058]],
        );
        assert_close(&g.get(&model.bias).as_array(), &[0.40265593, -0.2874091]);
    }

    // #[test]
    // fn test_linear_missing_gradients() {
    //     let mut model: Linear<5, 3> = Default::default();
    //     let mut g: SimpleGradients = Default::default();

    //     // no gradients present
    //     let mut unused = Default::default();
    //     model.update(&mut g, &mut unused);
    //     assert_eq!(&unused.ids, &[*model.weight.id(), *model.bias.id()]);

    //     g.0.mut_gradient(&model.weight);

    //     // weight gradient is present
    //     let mut unused = Default::default();
    //     model.update(&mut g, &mut unused);
    //     assert_eq!(&unused.ids, &[*model.bias.id()]);

    //     g.0.mut_gradient(&model.weight);
    //     g.0.mut_gradient(&model.bias);

    //     // both gradients present
    //     let mut unused = Default::default();
    //     model.update(&mut g, &mut unused);
    //     assert!(unused.is_empty());
    // }
}
