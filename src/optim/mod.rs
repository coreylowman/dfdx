//! Optimizers such as [Sgd], [Adam], and [RMSprop] that can optimize neural networks.
//!
//! # Initializing
//!
//! All the optimizer's provide [Default] implementations, and also provide a way to specify
//! all the relevant parameters through the corresponding config object:
//! - [Sgd::new()] with [SgdConfig]
//! - [Adam::new()] with [AdamConfig]
//! - [RMSprop::new()] with [RMSpropConfig]
//!
//! # Updating network parameters
//!
//! This is done via [Optimizer::update()], where you pass in a mutable [crate::nn::Module], and
//! the [crate::tensor::Gradients]:
//!
//! ```rust
//! # use dfdx::{prelude::*, optim::*, losses};
//! # type MyModel = Linear<5, 2>;
//! # let dev: Cpu = Default::default();
//! let mut model = MyModel::build_on_device(&dev);
//! let mut grads = model.alloc_grads();
//! let mut opt = Sgd::new(&model, Default::default());
//! # let x: Tensor<Rank1<5>, f32, _> = dev.zeros();
//! # let y = model.forward(x.traced(grads));
//! # let loss = losses::mse_loss(y, dev.zeros());
//! // -- snip loss computation --
//!
//! grads = loss.backward();
//! opt.update(&mut model, &grads);
//! model.zero_grads(&mut grads);
//! ```

mod adam;
mod optimizer;
mod rmsprop;
mod sgd;

pub use adam::{Adam, AdamConfig, AdamKernel};
pub use optimizer::{Momentum, WeightDecay};
pub use optimizer::{Optimizer, OptimizerUpdateError, UnusedTensors};
pub use rmsprop::{RMSprop, RMSpropConfig, RMSpropKernel};
pub use sgd::{Sgd, SgdConfig, SgdKernel};

pub mod prelude {
    pub use super::{Optimizer, OptimizerUpdateError, UnusedTensors};
}
