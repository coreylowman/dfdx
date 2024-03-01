use crate::prelude::*;

/// An embedding
///
/// **Pytorch Equivalent**: `torch.nn.Embedding(...)`
///
/// Initializes embedding matrix from the Standard Normal distribution.
///
/// Generics:
/// - `Vocab`: The size of the vocabulary, inputs integer values must be between
///    0 and Vocab;
/// - `Model`: The "output" size of vectors & matrices which are the vectors being selected.
///
/// # Examples
/// `Embedding<5, 2>` can act on vectors with SEQ integer elements (with values between 0 and 4), and results in a SEQ tensor of
/// usually f32 elements being the rows in the embedding matrix.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = EmbeddingConstConfig<7, 2>;
/// let mut model = dev.build_module::<f32>(Model::default());
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
#[derive(Default, Clone, Copy, Debug)]
pub struct EmbeddingConfig<Vocab: Dim, Model: Dim> {
    pub vocab: Vocab,
    pub model: Model,
}

/// Compile time sugar alias around [EmbeddingConfig].
pub type EmbeddingConstConfig<const VOCAB: usize, const MODEL: usize> =
    EmbeddingConfig<Const<VOCAB>, Const<MODEL>>;

impl<V: Dim, M: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for EmbeddingConfig<V, M> {
    type Built = Embedding<V, M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(Embedding {
            weight: device.try_zeros_like(&(self.vocab, self.model))?,
        })
    }
}

/// See [EmbeddingConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Embedding<Vocab: Dim, Model: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub weight: Tensor<(Vocab, Model), Elem, Dev>,
}

impl<V: Dim, M: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for Embedding<V, M, E, D>
where
    rand_distr::StandardNormal: rand_distr::Distribution<E>,
{
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.weight.try_fill_with_distr(rand_distr::StandardNormal)
    }
}

impl<V: Dim, M: Dim, Seq: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Seq,), usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<(Seq, M), E, D, T>;

    fn try_forward(
        &self,
        input: Tensor<(Seq,), usize, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}

impl<Batch: Dim, Seq: Dim, V: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq), usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;

    fn try_forward(
        &self,
        input: Tensor<(Batch, Seq), usize, D, T>,
    ) -> Result<Self::Output, crate::tensor::Error> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    const W: [[f64; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];

    #[test]
    fn embedding_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = Embedding {
            weight: dev.tensor(W).to_dtype::<TestDtype>(),
        };

        let x = dev.tensor([0, 0, 1]);
        let y = model.forward(x.leaky_trace());
        assert_close_to_literal!(
            y,
            [
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
            ]
        );

        let g = y.square().mean().backward();
        assert_close_to_literal!(
            g.get(&model.weight),
            [
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
            ]
        );
    }
}
