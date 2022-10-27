/// L2 and decoupled regularization methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightDecay {
    /// Weight decay applied to the gradients before any momentum updates. Equivalent to L2 regularization.
    L2(f32),

    /// Weight decay applied after any momentum updates, without modifying the gradients.
    /// See [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    Decoupled(f32),
}
