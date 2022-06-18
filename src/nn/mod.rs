//! High level neural network building blocks such as [Linear], activations, and tuples as [Module]s.
//! Also includes `.save()` & `.load()` for all [Module]s.
//!
//! Saving and loading model parameters is done using [SaveToNpz] and [LoadFromNpz]. All modules provided here implement it,
//! including tuples. So you can call [SaveToNpz::save()] to save the module to a `.npz` zip file
//! format, and then [LoadFromNpz::load()] to load the weights.
//!
//! Randomizing parameters is done using [ResetParams::reset_params()]. All modules implement the underlying
//! logic themselves. For example [Linear] calculates the distribution it draws values from based on its input
//! size.

mod activations;
mod dropout;
mod impl_module_for_tuples;
mod layer_norm;
mod linear;
mod module;
mod npz;
mod repeated;
mod residual;

pub use activations::*;
pub use dropout::*;
pub use impl_module_for_tuples::*;
pub use layer_norm::*;
pub use linear::*;
pub use module::*;
pub use npz::*;
pub use repeated::*;
pub use residual::*;
