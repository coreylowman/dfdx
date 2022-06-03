//! High level neural network building blocks such as [Linear], activations, and tuples as feedforward networks.
//!
//! Saving model parameters is done using [SaveToZip]. All modules provided here implement it,
//! including tuples. So you can call [SaveToZip::save()] to save the module to a `.npz` zip file
//! format.

mod activations;
mod dropout;
mod impl_module_for_tuples;
mod linear;
mod module;

pub use activations::*;
pub use dropout::*;
pub use impl_module_for_tuples::*;
pub use linear::*;
pub use module::*;
