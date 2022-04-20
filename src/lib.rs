pub mod diff_fns;
pub mod gradients;
pub mod losses;
pub mod nn;
pub mod tensor;
pub mod tensor_ops;
pub mod prelude {
    pub use crate::diff_fns::*;
    pub use crate::gradients::GradientTape;
    pub use crate::losses::*;
    pub use crate::nn::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
}
