use dfdx::{
    prelude::{Device, Dim, Dtype, Tape, Tensor},
    shapes::Const,
    tensor::{PutTape, SplitTape},
    tensor_ops::GatherTo,
};

use crate::*;

#[derive(Default, Clone, Copy, Debug)]
pub struct EmbeddingConfig<Vocab: Dim, Model: Dim> {
    pub vocab: Vocab,
    pub model: Model,
}

pub type EmbeddingConstConfig<const VOCAB: usize, const MODEL: usize> =
    EmbeddingConfig<Const<VOCAB>, Const<MODEL>>;

impl<V: Dim, M: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for EmbeddingConfig<V, M> {
    type Built = Embedding<V, M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(Embedding {
            weight: device.try_zeros_like(&(self.vocab, self.model))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct Embedding<Vocab: Dim, Model: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub weight: Tensor<(Vocab, Model), Elem, Dev>,
}

impl<V: Dim, M: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for Embedding<V, M, E, D>
where
    rand_distr::StandardNormal: rand_distr::Distribution<E>,
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.weight.try_fill_with_distr(rand_distr::StandardNormal)
    }
}

impl<V: Dim, M: Dim, Seq: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Seq,), usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<(Seq, M), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(Seq,), usize, D, T>) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}

impl<Batch: Dim, Seq: Dim, V: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq), usize, D, T>> for Embedding<V, M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Batch, Seq), usize, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        self.weight.clone().put_tape(tape).try_gather(input)
    }
}
