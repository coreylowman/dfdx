use num_traits::Float;
use rand_distr::{uniform::SampleUniform, Uniform};

use crate::{shapes::*, tensor::*, tensor_ops::*};

use super::*;

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
/// let mut model = dev.build_module::<Model, f32>();
/// // single sequence of ids
/// let inputs: Tensor<Rank1<5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<5>, Const<2>,), f32, _> = model.forward(inputs);
/// // Dynamic sequence of ids
/// let inputs: Tensor<(usize, ), usize, _> = dev.zeros_like(&(5, ));
/// let _: Tensor<(usize, Const<2>,), f32, _> = model.forward(inputs);
/// // batched sequence of ids
/// let inputs: Tensor<Rank2<10, 5>, usize, _> = dev.zeros();
/// let _: Tensor<(Const<10>, Const<5>, Const<2>), f32, _> = model.forward(inputs);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding<const VOCAB: usize, const DIM: usize, E: Dtype, D: DeviceStorage> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<VOCAB, DIM>, E, D>,
}

impl<const V: usize, const M: usize, E: Dtype, D: DeviceStorage> NonMutableModule
    for Embedding<V, M, E, D>
{
}

impl<const C: usize, const M: usize, E: Dtype + Float + SampleUniform, D: Device<E>>
    TensorCollection<E, D> for Embedding<C, M, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = Embedding<C, M, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor(
                "weight",
                |s| &s.weight,
                |s| &mut s.weight,
                TensorOptions::reset_with(|t| {
                    let b: E = E::ONE / E::from_usize(C).unwrap().sqrt();
                    t.try_fill_with_distr(Uniform::new(-b, b))
                }),
            ),
            |weight| Embedding { weight },
        )
    }
}

impl<const V: usize, const M: usize, SEQ: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(SEQ,), usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<(SEQ, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(SEQ,), usize, D, T>) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        BATCH: Dim,
        SEQ: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(BATCH, SEQ), usize, D, T>> for Embedding<VOCAB, DIM, E, D>
{
    type Output = Tensor<(BATCH, SEQ, Const<DIM>), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(BATCH, SEQ), usize, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

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
        let y = model.forward(x.leaky_trace());
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
        let y = model.forward(x.leaky_trace());
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
}
