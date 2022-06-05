//! High level neural network building blocks such as [Linear], activations, and tuples as feedforward networks.
//!
//! Saving and loading model parameters is done using [SaveToNpz] and [LoadFromNpz]. All modules provided here implement it,
//! including tuples. So you can call [SaveToNpz::save()] to save the module to a `.npz` zip file
//! format, and then [LoadFromNpz::load()] to load the weights.

mod activations;
mod dropout;
mod impl_module_for_tuples;
mod linear;
mod module;
mod npz;

pub use activations::*;
pub use dropout::*;
pub use impl_module_for_tuples::*;
pub use linear::*;
pub use module::*;
pub use npz::*;
