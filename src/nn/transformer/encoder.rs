use crate::prelude::*;

/// **Requires Nightly** A single transformer encoder block
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `K` The size of the keys and queries in the self attention layer.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
pub type TransformerEncoderBlock<const M: usize, const I: usize, const K: usize, const H: usize> = (
    Residual<MultiHeadAttention<M, M, K, M, H>>,
    LayerNorm1D<M>,
    Residual<(Linear<M, I>, ReLU, Linear<I, M>)>,
    LayerNorm1D<M>,
);

/// **Requires Nightly** A transformer encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
pub type TransformerEncoder<const M: usize, const I: usize, const L: usize, const H: usize> =
    Repeated<TransformerEncoderBlock<M, I, M, H>, L>;
