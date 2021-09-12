#![feature(generic_associated_types)]

pub mod gradients;
pub mod nn;
pub mod optim;
pub mod randomize;
pub mod tensor;

mod macros;
mod prelude;
pub use crate::macros::*;
