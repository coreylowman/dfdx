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
//! the [crate::gradients::Gradients]:
//!
//! ```rust
//! # use dfdx::{prelude::*, optim::*, losses, gradients::Gradients};
//! # type MyModel = Linear<5, 2>;
//! # let dev: Cpu = Default::default();
//! let mut model = MyModel::build_on_device(&dev);
//! let mut opt: Sgd<MyModel> = Default::default();
//! # let y = model.forward(dev.zeros::<Rank1<5>>().traced());
//! # let loss = losses::mse_loss(y, dev.zeros());
//! // -- snip loss computation --
//!
//! let gradients: Gradients = loss.backward();
//! opt.update(&mut model, gradients);
//! ```

mod adam;
mod optimizer;
mod rmsprop;
mod sgd;

pub use adam::{Adam, AdamConfig};
pub use optimizer::{GradientUpdate, Optimizer, OptimizerUpdateError, ParamUpdater, UnusedTensors};
pub use optimizer::{Momentum, WeightDecay};
pub use rmsprop::{RMSprop, RMSpropConfig};
pub use sgd::{Sgd, SgdConfig};

pub mod prelude {
    pub use super::{GradientUpdate, Optimizer, OptimizerUpdateError, ParamUpdater, UnusedTensors};
}
