use crate::prelude::*;

/// **Requires Nightly** A single transformer encoder block
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in the feedforward network.
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
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerEncoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerEncoderBlock] to use.
/// TODO: Doctests
pub type TransformerEncoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
> = Repeated<TransformerEncoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>, NUM_LAYERS>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_block_forward() {}

    #[test]
    fn test_encoder_forward() {}
}
