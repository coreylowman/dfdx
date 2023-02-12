use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{gradients::Tape, optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::module::{BuildModule, BuildOnDevice, Module, ModuleMut, ResetParams, ToDevice};

pub mod builder {
    #[derive(Debug)]
    pub struct Embedding<const VOCAB: usize, const DIM: usize>;
}

impl<const V: usize, const M: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E>
    for builder::Embedding<V, M>
where
    Embedding<V, M, E, D>: BuildModule<D, E>,
{
    type Built = Embedding<V, M, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

/// An embedding
/// Initializes [Self::weight] from a Uniform distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// # Generics
/// - `VOCAB` The size of the vocabulary, inputs integer values must be between
///    0 and VOCAB;
/// - `DIM` The "output" size of vectors & matrices which are the vectors being selected.
///
/// # Examples
/// `Embedding<5, 2>` can act on vectors with SEQ integer elements (with values between 0 and 4), and results in a SEQ tensor of
/// usually f32 elements being the rows in [Self::weight].
/// ```rust
///
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Embedding<7, 2>;
/// let mut model = Model::build_on_device(&dev);
/// // single sequence of ids
/// let inputs: Tensor<Rank1<5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<5>, Const<2>,), f32, _> = model.forward(inputs);
/// // batched sequence of ids
/// let inputs: Tensor<Rank2<10, 5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<10>, Const<5>, Const<2>), f32, _> = model.forward(inputs);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding<const VOCAB: usize, const DIM: usize, E: Dtype, D: DeviceStorage> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<VOCAB, DIM>, E, D>,
}

impl<const V: usize, const M: usize, const S: usize, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<Rank1<S>, usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<Rank2<S, M>, E, D, T>;
    fn forward(&self, input: Tensor<Rank1<S>, usize, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).gather(input)
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        const SEQ: usize,
        const BATCH: usize,
        E: Dtype,
        D: Device<E>,
        T: Tape<D>,
    > Module<Tensor<Rank2<BATCH, SEQ>, usize, D, T>> for Embedding<VOCAB, DIM, E, D>
{
    type Output = Tensor<Rank3<BATCH, SEQ, DIM>, E, D, T>;
    fn forward(&self, input: Tensor<Rank2<BATCH, SEQ>, usize, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).gather(input)
    }
}

impl<T, const VOCAB: usize, const DIM: usize, E: Dtype, D: Device<E>> ModuleMut<T>
    for Embedding<VOCAB, DIM, E, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

impl<const VOCAB: usize, const DIM: usize, E: Dtype, D: Device<E>> GradientUpdate<D, E>
    for Embedding<VOCAB, DIM, E, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, E>,
    {
        self.weight.update(updater, unused)?;
        Ok(())
    }
}

impl<const V: usize, const M: usize, E: Dtype + Float + SampleUniform, D: Device<E>>
    BuildModule<D, E> for Embedding<V, M, E, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound = E::ONE / E::from_usize(V).unwrap().sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        let weight = device.try_sample(distr)?;
        Ok(Self { weight })
    }
}

impl<const VOCAB: usize, const DIM: usize, E: Dtype + Float + SampleUniform, D: Device<E>>
    ResetParams<D, E> for Embedding<VOCAB, DIM, E, D>
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let bound = E::ONE / E::from_usize(VOCAB).unwrap().sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        self.weight.try_fill_with_distr(distr)?;
        Ok(())
    }
}

impl<const VOCAB: usize, const DIM: usize, E: Dtype, D1: Device<E>, D2: Device<E>> ToDevice<D2>
    for Embedding<VOCAB, DIM, E, D1>
{
    type Output = Embedding<VOCAB, DIM, E, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        Embedding {
            weight: self.weight.to_device(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{tests::SimpleUpdater, DeviceBuildExt},
        tests::*,
        unique_id::HasUniqueId,
    };

    const W: [[TestDtype; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];

    #[test]
    fn test_embedding_initialize() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<builder::Embedding<2000, 1>, TestDtype>();
        let bound = 1.0 / (2000.0.sqrt());
        for v in m.weight.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
    }

    #[test]
    fn embedding_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = Embedding {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([0, 0, 1]);
        let y = model.forward(x.trace());
        assert_close(
            &y.array(),
            &[
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
            ],
        );

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [
                    -0.09223715,
                    -0.08099073,
                    -0.09898819,
                    0.03814289,
                    -0.007172427,
                ],
                [
                    0.015645266,
                    0.01874625,
                    -0.014227235,
                    -0.012497525,
                    0.025299065,
                ],
            ],
        );
    }

    #[test]
    fn test_forward_2d() {
        let dev: TestDevice = Default::default();

        let model = Embedding {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([[0, 0], [0, 1]]);
        let y = model.forward(x.trace());
        assert_close(
            &y.array(),
            &[
                [
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                ],
                [
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                    [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
                ],
            ],
        );

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [
                    -0.103766784,
                    -0.091114566,
                    -0.11136171,
                    0.042910747,
                    -0.008068981,
                ],
                [
                    0.011733949,
                    0.014059687,
                    -0.010670426,
                    -0.009373143,
                    0.018974299,
                ],
            ],
        );
    }

    #[test]
    fn test_embedding_missing_gradients() {
        let dev: TestDevice = Default::default();

        let mut model = dev.build_module::<builder::Embedding<5, 3>, TestDtype>();
        let mut g: SimpleUpdater = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(&unused.ids, &[*model.weight.id()]);

        g.0.try_alloc_for(&model.weight).unwrap();

        // weight gradient is present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
