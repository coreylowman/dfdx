use crate::prelude::*;

/// **Requires Nightly** A single transformer encoder block
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `K` The size of the keys and queries in the self attention layer.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
pub type TransformerEncoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
> = (
    Residual<MultiHeadAttention<MODEL_DIM, NUM_HEADS>>,
    LayerNorm1D<MODEL_DIM>,
    Residual<(Linear<MODEL_DIM, FF_DIM>, ReLU, Linear<FF_DIM, MODEL_DIM>)>,
    LayerNorm1D<MODEL_DIM>,
);

/// **Requires Nightly** A transformer encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
pub type TransformerEncoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
> = Repeated<TransformerEncoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>, NUM_LAYERS>;
