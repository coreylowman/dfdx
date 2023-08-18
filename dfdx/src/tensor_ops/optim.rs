/// L2 and decoupled regularization methods
#[derive(Debug, Clone, Copy)]
pub enum WeightDecay {
    /// Weight decay applied to the gradients before any momentum updates. Equivalent to L2 regularization.
    L2(f64),

    /// Weight decay applied after any momentum updates, without modifying the gradients.
    /// See [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    Decoupled(f64),
}

/// Used to communicate the "WeightDecay" enum to cuda kernels
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) enum WeightDecayType {
    None,
    L2,
    Decoupled,
}

#[cfg(feature = "cuda")]
pub(super) fn weight_decay_to_cuda(wd: Option<WeightDecay>) -> (WeightDecayType, f64) {
    match wd {
        None => (WeightDecayType::None, Default::default()),
        Some(WeightDecay::L2(x)) => (WeightDecayType::L2, x),
        Some(WeightDecay::Decoupled(x)) => (WeightDecayType::Decoupled, x),
    }
}

/// Momentum used for [crate::optim::Sgd] and others
#[derive(Debug, Clone, Copy)]
pub enum Momentum {
    /// Momentum that is applied to the velocity of a parameter directly.
    Classic(f64),

    /// Momentum that is applied to both velocity and gradients. See [crate::optim::Sgd] nesterov paper for more.
    Nesterov(f64),
}

/// Used to communicate the "Momentum" enum to cuda kernels
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) enum MomentumType {
    None,
    Classic,
    Nesterov,
}

#[cfg(feature = "cuda")]
pub(super) fn momentum_to_cuda(wd: Option<Momentum>) -> (MomentumType, f64) {
    match wd {
        None => (MomentumType::None, Default::default()),
        Some(Momentum::Classic(x)) => (MomentumType::Classic, x),
        Some(Momentum::Nesterov(x)) => (MomentumType::Nesterov, x),
    }
}
