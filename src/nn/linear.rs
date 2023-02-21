use crate::{gradients::Tape, shapes::*, tensor::*, tensor_ops::*};

use super::{
    module::{BuildModule, BuildOnDevice, Module, ModuleMut, ToDevice},
    TensorFunction, TensorFunctionOption, TensorVisitor, VisitTensors,
};

use num_traits::Float;
use rand_distr::{uniform::SampleUniform, Uniform};

pub mod builder {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct Linear<const I: usize, const O: usize>;
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E>
    for builder::Linear<I, O>
where
    Linear<I, O, E, D>: BuildModule<D, E>,
{
    type Built = Linear<I, O, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// A linear transformation of the form `weight * x + bias`, where `weight` is a matrix, `x` is a vector or matrix,
/// and `bias` is a vector.
///
/// Initializes [Self::weight] and [Self::bias] from a Uniform distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// # Generics
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// # Examples
/// `Linear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Linear<5, 2>;
/// let model = dev.build_module::<Model, f32>();
/// // single item forward
/// let _: Tensor<Rank1<2>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// // batched forward
/// let _: Tensor<Rank2<10, 2>, f32, _> = model.forward(dev.zeros::<Rank2<10, 5>>());
/// ```
#[derive(Debug, Clone)]
pub struct Linear<const I: usize, const O: usize, E: Dtype, D: DeviceStorage> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<O, I>, E, D>,

    /// Bias vector, shape (O, )
    pub bias: Tensor<Rank1<O>, E, D>,
}

impl<
        const I: usize,
        const O: usize,
        E: Dtype,
        D: DeviceStorage,
    > VisitTensors<E, D> for Linear<I, O, E, D>
{
    fn visit_groups<const N: usize, const M: usize, F: TensorFunction<N, M, E, D>>(
        mut visitor: TensorVisitor<N, M, Self, F>,
    ) -> Result<(), F::Err> {
        let b = 1. / (I as f64).sqrt();
        let options = [TensorFunctionOption::ResetParamsUniform(-b, b)];

        visitor.visit_field_with_options(|s| &s.weight, |s| &mut s.weight, "weight", &options)?;
        visitor.visit_field_with_options(|s| &s.bias, |s| &mut s.bias, "bias", &options)
    }
}

impl<const I: usize, const O: usize, E: Dtype + Float + SampleUniform, D: Device<E>>
    BuildModule<D, E> for Linear<I, O, E, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let b: E = E::ONE / E::from_usize(I).unwrap().sqrt();
        let weight = device.try_sample(Uniform::new(-b, b))?;
        let bias = device.try_sample(Uniform::new(-b, b))?;
        Ok(Self { weight, bias })
    }
}

impl<const I: usize, const O: usize, E: Dtype, D1: Device<E>, D2: Device<E>> ToDevice<D2>
    for Linear<I, O, E, D1>
{
    type Output = Linear<I, O, E, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        Linear {
            weight: self.weight.to_device(device),
            bias: self.bias.to_device(device),
        }
    }
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T> Module<T> for Linear<I, O, E, D>
where
    T: SplitTape + TryMatMul<Tensor<Rank2<I, O>, E, D, T::Tape>>,
    T::Tape: Tape<D>,
    for<'a> Bias1D<'a, O, E, D>: Module<T::Output, Output = T::Output>,
{
    type Output = T::Output;

    /// 1d forward using [matmul()] and [add()].
    fn forward(&self, x: T) -> Self::Output {
        let o = x.matmul(self.weight.retaped::<T::Tape>().permute());
        Bias1D { beta: &self.bias }.forward(o)
    }
}

impl<T, const I: usize, const O: usize, E: Dtype, D: Device<E>> ModuleMut<T> for Linear<I, O, E, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[derive(Clone, Debug)]
struct Bias1D<'a, const M: usize, E: Dtype, D: DeviceStorage> {
    beta: &'a Tensor<Rank1<M>, E, D>,
}

impl<'a, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>> Module<Tensor<Rank1<M>, E, D, T>>
    for Bias1D<'a, M, E, D>
{
    type Output = Tensor<Rank1<M>, E, D, T>;
    fn forward(&self, input: Tensor<Rank1<M>, E, D, T>) -> Self::Output {
        input + self.beta.clone()
    }
}

impl<'a, B: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(B, Const<M>), E, D, T>> for Bias1D<'a, M, E, D>
{
    type Output = Tensor<(B, Const<M>), E, D, T>;
    fn forward(&self, input: Tensor<(B, Const<M>), E, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

impl<'a, B: Dim, S: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(B, S, Const<M>), E, D, T>> for Bias1D<'a, M, E, D>
{
    type Output = Tensor<(B, S, Const<M>), E, D, T>;
    fn forward(&self, input: Tensor<(B, S, Const<M>), E, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{tests::SimpleUpdater, DeviceBuildExt},
        optim::GradientUpdate,
        tests::*,
        unique_id::HasUniqueId,
    };

    const W: [[TestDtype; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [TestDtype; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_linear_ondevice() {
        let dev: TestDevice = Default::default();
        let _: Linear<1, 1, TestDtype, _> = BuildModule::build(&dev);
        let _: Linear<1, 1, TestDtype, TestDevice> = builder::Linear::<1, 1>::build_on_device(&dev);
        let _: Linear<1, 1, TestDtype, _> = builder::Linear::<1, 1>::build_on_device(&dev);
        let _ = dev.build_module::<builder::Linear<1, 1>, TestDtype>();
    }

    #[test]
    fn test_linear_initialize() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<builder::Linear<2000, 1>, TestDtype>();
        let bound: TestDtype = 1.0 / 2000.0;
        let bound = bound.sqrt();
        for v in m.weight.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
        for v in m.bias.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
    }

    #[test]
    fn test_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = Linear {
            weight: dev.tensor(W),
            bias: dev.tensor(B),
        };

        let x = dev.tensor([-0.8808001, 2.4185333, 2.2478335, 0.0565211, 2.031299]);
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
        let dev: TestDevice = Default::default();

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
        let dev: TestDevice = Default::default();

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
        let dev: TestDevice = Default::default();

        let mut model = dev.build_module::<builder::Linear<5, 3>, TestDtype>();
        let mut g: SimpleUpdater = Default::default();

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
