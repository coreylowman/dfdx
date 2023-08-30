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
//! This is done via [crate::Optimizer::update()], where you pass in a mutable [crate::Module], and
//! the [dfdx::tensor::Gradients]:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # use dfdx_nn::*;
//! # let dev: Cpu = Default::default();
//! use dfdx_nn::optim::*;
//! type Model = LinearConstConfig<5, 2>;
//! let mut model = dev.build_module::<f32>(Model::default());
//! let mut grads = model.alloc_grads();
//! let mut opt = Sgd::new(&model, Default::default());
//! # let x: Tensor<Rank1<5>, f32, _> = dev.zeros();
//! # let y = model.forward(x.traced(grads));
//! # let loss = dfdx::losses::mse_loss(y, dev.zeros());
//! // -- snip loss computation --
//!
//! grads = loss.backward();
//! opt.update(&mut model, &grads);
//! model.zero_grads(&mut grads);
//! ```

mod adam;
mod rmsprop;
mod sgd;

pub use adam::Adam;
pub use rmsprop::RMSprop;
pub use sgd::Sgd;
// re-exports
pub use dfdx::tensor_ops::{AdamConfig, Momentum, RMSpropConfig, SgdConfig, WeightDecay};
