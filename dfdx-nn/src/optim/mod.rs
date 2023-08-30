mod adam;
mod rmsprop;
mod sgd;

pub use adam::Adam;
pub use rmsprop::RMSprop;
pub use sgd::Sgd;
// re-exports
pub use dfdx::tensor_ops::{AdamConfig, Momentum, RMSpropConfig, SgdConfig, WeightDecay};
