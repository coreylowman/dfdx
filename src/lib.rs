pub mod diff_fns;
pub mod gradients;
pub mod nn;
pub mod tensor;
pub mod prelude {
    pub use crate::diff_fns::*;
    pub use crate::gradients::GradientTape;
    pub use crate::nn::*;
    pub use crate::tensor::*;
}
