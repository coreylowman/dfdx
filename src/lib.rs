//! Ergonomics & safety focused deep learning in Rust. Main features include:
//! 1. Tensor library, complete with const generic shapes, activation functions, and more.
//! 2. Safe & Easy to use neural network building blocks.
//! 3. Standard deep learning optimizers such as Sgd and Adam.
//! 4. Reverse mode auto differentiation[1] implementation.

pub mod arrays;
pub mod data;
pub mod devices;
pub mod gradients;
pub mod losses;
pub mod nn;
pub mod numpy;
pub mod optim;
pub mod tensor;
pub mod tensor_ops;
pub mod unique_id;

/// Contains all public exports.
pub mod prelude {
    pub use crate::arrays::*;
    pub use crate::data::*;
    pub use crate::devices::*;
    pub use crate::gradients::*;
    pub use crate::losses::*;
    pub use crate::nn::*;
    pub use crate::optim::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
    pub use crate::unique_id::*;
}
