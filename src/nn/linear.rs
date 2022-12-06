use crate::{gradients::Tape, optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::module::{BuildModule, Module, ModuleMut};

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
///
/// Initializes [Self::weight] and [Self::bias] from a [Uniform] distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// This uses [Randomize::randomize()] to set the values of the tensor.
#[derive(Debug, Clone)]
pub struct Linear<const I: usize, const O: usize, D: Device<f32> = Cpu> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<O, I>, f32, D>,

    /// Bias vector, shape (O, )
    pub bias: Tensor<Rank1<O>, f32, D>,
}

impl<const I: usize, const O: usize, D: Device<f32>> GradientUpdate<D, f32> for Linear<I, O, D> {
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.weight.update(updater, unused)?;
        self.bias.update(updater, unused)?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, D: Device<f32>> BuildModule<D, f32> for Linear<I, O, D> {
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound: f32 = 1.0 / (I as f32).sqrt();
        let weight = device.try_uniform(-bound, bound)?;
        let bias = device.try_uniform(-bound, bound)?;
        Ok(Self { weight, bias })
    }

    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let bound: f32 = 1.0 / (I as f32).sqrt();
        self.weight.try_fill_with_uniform(-bound, bound)?;
        self.bias.try_fill_with_uniform(-bound, bound)?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, D: Device<f32>, T> Module<T> for Linear<I, O, D>
where
    T: SplitTape + TryMatMul<Tensor<Rank2<I, O>, f32, D, T::Tape>>,
    T::Tape: Tape<D>,
    for<'a> Bias1D<'a, O, D>: Module<T::Output, Output = T::Output>,
{
    type Output = T::Output;

    /// 1d forward using [vecmat_mul()] and [add()].
    fn forward(&self, x: T) -> Self::Output {
        let o = x.matmul(self.weight.retaped::<T::Tape>().permute());
        Bias1D { beta: &self.bias }.forward(o)
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

#[derive(Clone, Debug)]
struct Bias1D<'a, const M: usize, D: Device<f32> = Cpu> {
    beta: &'a Tensor<Rank1<M>, f32, D>,
}

impl<'a, const M: usize, D: Device<f32>, T: Tape<D>> Module<Tensor<Rank1<M>, f32, D, T>>
    for Bias1D<'a, M, D>
{
    type Output = Tensor<Rank1<M>, f32, D, T>;
    fn forward(&self, input: Tensor<Rank1<M>, f32, D, T>) -> Self::Output {
        input + self.beta.clone()
    }
}

impl<'a, B: Dim, const M: usize, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(B, Const<M>), f32, D, T>> for Bias1D<'a, M, D>
{
    type Output = Tensor<(B, Const<M>), f32, D, T>;
    fn forward(&self, input: Tensor<(B, Const<M>), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

impl<'a, B: Dim, S: Dim, const M: usize, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(B, S, Const<M>), f32, D, T>> for Bias1D<'a, M, D>
{
    type Output = Tensor<(B, S, Const<M>), f32, D, T>;
    fn forward(&self, input: Tensor<(B, S, Const<M>), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::tests::SimpleUpdater,
        tests::{assert_close, build_test_device},
        unique_id::HasUniqueId,
    };

    const W: [[f32; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f32; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_linear_initialize() {
        let dev = build_test_device!();
        let m: Linear<2000, 1, _> = BuildModule::build(&dev);
        let bound = 1.0 / 2000.0f32.sqrt();
        for v in m.weight.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
        for v in m.bias.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
    }

    #[test]
    fn test_forward_1d() {
        let dev = build_test_device!();

        let model = Linear {
            weight: dev.tensor(W),
            bias: dev.tensor(B),
        };

        let x = dev.tensor([-0.8808001f32, 2.4185333, 2.2478335, 0.0565211, 2.031299]);
        let y = model.forward(x.trace());
        assert_close(&y.array(), &[-0.93430865, 0.08624211]);

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [0.82293916, -2.2596567, -2.1001704, -0.05280815, -1.8978603],
                [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
            ],
        );
        assert_close(&g.get(&model.bias).array(), &[-0.93430865, 0.08624211]);
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
            &y.array(),
            &[
                [1.3914378, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872],
            ],
        );

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [-1.1541969, 0.6956873, -0.8553807, 0.9289255, 0.04931633],
                [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.5286576],
            ],
        );
        assert_close(&g.get(&model.bias).array(), &[0.7679174, -0.31687993]);
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
            &y.array(),
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
            &g.get(&model.weight).array(),
            &[[-0.16088384, 0.10978711, -0.9008978, 0.59211355, -0.029177088], [0.35563633, -0.38838047, -0.17600831, -0.2034213, 0.31128058]],
        );
        assert_close(&g.get(&model.bias).array(), &[0.40265593, -0.2874091]);
    }

    #[test]
    fn test_linear_missing_gradients() {
        let dev = build_test_device!();

        let mut model: Linear<5, 3, _> = BuildModule::build(&dev);
        let mut g: SimpleUpdater<_> = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(&unused.ids, &[*model.weight.id(), *model.bias.id()]);

        g.0.try_alloc_for(&model.weight).unwrap();

        // weight gradient is present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(&unused.ids, &[*model.bias.id()]);

        g.0.try_alloc_for(&model.weight).unwrap();
        g.0.try_alloc_for(&model.bias).unwrap();

        // both gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
