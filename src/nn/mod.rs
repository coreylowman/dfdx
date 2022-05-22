//! Neural network building blocks such as [Linear] and impls for tuples as feedforward networks.

mod impl_module_for_activations;
mod impl_module_for_tuples;
mod linear;
mod module;

pub use impl_module_for_activations::*;
pub use impl_module_for_tuples::*;
pub use linear::*;
pub use module::*;
