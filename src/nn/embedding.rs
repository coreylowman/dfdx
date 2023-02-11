use crate::{gradients::Tape, optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::module::{BuildModule, BuildOnDevice, Module, ModuleMut, ResetParams, ToDevice};

pub mod builder {
    #[derive(Debug)]
    pub struct Embedding<const VOCAB: usize, const DIM: usize>;

    #[derive(Debug)]
    pub struct LearnedPositionalEmbedding<const MAX_LEN: usize, const DIM: usize>;
}

impl<const V: usize, const M: usize, D: Device<f32>> BuildOnDevice<D, f32>
    for builder::Embedding<V, M>
{
    type Built = Embedding<V, M, f32, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

impl<const V: usize, const M: usize, D: Device<f32>> BuildOnDevice<D, f32>
    for builder::LearnedPositionalEmbedding<V, M>
{
    type Built = LearnedPositionalEmbedding<V, M, f32, D>;
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

impl<const VOCAB: usize, const DIM: usize, SEQ: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(SEQ,), usize, D, T>> for Embedding<VOCAB, DIM, f32, D>
{
    type Output = Tensor<(SEQ, Const<DIM>), f32, D, T>;
    fn forward(&self, input: Tensor<(SEQ,), usize, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).gather(input)
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        SEQ: Dim,
        BATCH: Dim,
        D: Device<f32>,
        T: Tape<D>,
    > Module<Tensor<(BATCH, SEQ), usize, D, T>> for Embedding<VOCAB, DIM, f32, D>
{
    type Output = Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>;
    fn forward(&self, input: Tensor<(BATCH, SEQ), usize, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).gather(input)
    }
}

impl<T, const VOCAB: usize, const DIM: usize, D: Device<f32>> ModuleMut<T>
    for Embedding<VOCAB, DIM, f32, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

impl<const VOCAB: usize, const DIM: usize, D: Device<f32>> GradientUpdate<D, f32>
    for Embedding<VOCAB, DIM, f32, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.weight.update(updater, unused)?;
        Ok(())
    }
}

impl<const V: usize, const M: usize, D: Device<f32>> BuildModule<D, f32>
    for Embedding<V, M, f32, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound: f32 = 1.0 / (V as f32).sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        let weight = device.try_sample(distr)?;
        Ok(Self { weight })
    }
}

impl<const VOCAB: usize, const DIM: usize, D: Device<f32>> ResetParams<D, f32>
    for Embedding<VOCAB, DIM, f32, D>
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let bound: f32 = 1.0 / (VOCAB as f32).sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        self.weight.try_fill_with_distr(distr)?;
        Ok(())
    }
}

impl<const VOCAB: usize, const DIM: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2>
    for Embedding<VOCAB, DIM, f32, D1>
{
    type Output = Embedding<VOCAB, DIM, f32, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        Embedding {
            weight: self.weight.to_device(device),
        }
    }
}


/// A learned positional embedding
/// Initializes [Self::weight] from a Uniform distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// # Generics
/// - `MAX_LEN` The maximum length of the input sequences. This determines how many positional embeddings to train.
/// - `DIM` The "output" size of vectors & matrices which are the vectors being selected.
///
/// # Examples
/// `LearnedPositionalEmbedding<5, 2>` can act on vectors with SEQ x DIM elements (pre-embedded), 
/// resulting in output vectors of the same size with positional embeddings added.
/// 
/// ```rust
///
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = (Embedding<7, 2>, LearnedPositionalEmbedding<20, 2>);
/// let mut model = Model::build_on_device(&dev);
/// // single sequence of ids
/// let inputs: Tensor<Rank1<5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<5>, Const<2>,), f32, _> = model.forward(inputs);
/// // batched sequence of ids
/// let inputs: Tensor<Rank2<10, 5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<10>, Const<5>, Const<2>), f32, _> = model.forward(inputs);
/// ```
#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding<const MAX_LEN: usize, const DIM: usize, E: Dtype, D: DeviceStorage> {
    /// Learned positonal embeddings
    embed: Embedding<MAX_LEN, DIM, E, D>,
    device: D,
}

/// Pass in an unbatched pre-embedded sequence, add positional embeddings in
impl<const MAX_LEN: usize, const DIM: usize, SEQ: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(SEQ, Const<DIM>), f32, D, T>> for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
where
    D: TensorFrom<std::vec::Vec<usize>, (SEQ,), usize>
{
    type Output = Tensor<(SEQ, Const<DIM>), f32, D, T>;
    fn forward(&self, input: Tensor<(SEQ, Const<DIM>), f32, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        let positions: Tensor<(SEQ,), usize, D> = self.device.tensor((0..input.shape().0.size()).collect::<std::vec::Vec<_>>());
        let position_embeddings = self.embed.forward(positions.put_tape(tape));
        position_embeddings + input
    }
}

impl<
        const MAX_LEN: usize,
        const DIM: usize,
        SEQ: Dim,
        BATCH: Dim,
        D: Device<f32>,
        T: Tape<D>,
    > Module<Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>> for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
where
    D: TensorFrom<std::vec::Vec<usize>, (BATCH, SEQ,), usize>
{
    type Output = Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>;
    fn forward(&self, input: Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        let positions: Tensor<(BATCH, SEQ,), usize, D> = self.device.tensor((0..input.shape().1.size()).cycle().take(input.shape().1.size() * input.shape().0.size()).collect::<std::vec::Vec<_>>());
        let position_embeddings = self.embed.forward(positions.put_tape(tape));
        position_embeddings + input
    }
}

impl<T, const MAX_LEN: usize, const DIM: usize, D: Device<f32>> ModuleMut<T>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

impl<const MAX_LEN: usize, const DIM: usize, D: Device<f32>> GradientUpdate<D, f32>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.embed.update(updater, unused)
    }
}

impl<const MAX_LEN: usize, const DIM: usize, D: Device<f32>> BuildModule<D, f32>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        Ok(Self {embed: Embedding::try_build(device)?, device: device.clone()})
    }
}

impl<const MAX_LEN: usize, const DIM: usize, D: Device<f32>> ResetParams<D, f32>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.embed.try_reset_params()
    }
}

impl<const MAX_LEN: usize, const DIM: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D1>
{
    type Output = LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        LearnedPositionalEmbedding {
            embed: self.embed.to_device(device),
            device: device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::tests::SimpleUpdater,
        tests::{assert_close, TestDevice},
        unique_id::HasUniqueId,
    };

    const W: [[f32; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];

    #[test]
    fn test_embedding_initialize() {
        let dev: TestDevice = Default::default();
        let m = builder::Embedding::<2000, 1>::build_on_device(&dev);
        let bound = 1.0 / 2000.0f32.sqrt();
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

        let mut model = builder::Embedding::<5, 3>::build_on_device(&dev);
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
