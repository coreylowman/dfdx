//! Neural network building blocks such as [Linear] and impls for tuples as feedforward networks.

mod activations;
mod impl_module_for_tuples;
mod linear;
mod module;

pub use activations::*;
pub use impl_module_for_tuples::*;
pub use linear::*;
pub use module::*;
