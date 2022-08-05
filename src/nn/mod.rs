//! High level neural network building blocks such as [Linear], activations, and tuples as [Module]s.
//! Also includes `.save()` & `.load()` for all [Module]s.
//!
//! # Initializing
//!
//! All modules implement [Default], and this initializes all parameters to `0.0`. The intention is then
//! to call [ResetParams::reset_params()], which randomizes the parameters:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let mut rng = rand::thread_rng();
//! let mut model: Linear<5, 2> = Default::default(); // set all params to 0
//! model.reset_params(&mut rng); // randomize weights
//! ```
//!
//! # Sequential models
//!
//! Tuple's implement [Module], so you can string multiple module's together.
//!
//! Here's a single layer MLP:
//! ```rust
//! # use dfdx::prelude::*;
//! type Mlp = (Linear<5, 3>, ReLU, Linear<3, 2>);
//! ```
//!
//! Here's a more complex feedforward network that takes vectors of 5 elements and maps them to 2 elements.
//! ```rust
//! # use dfdx::prelude::*;
//! type ComplexNetwork = (
//!     DropoutOneIn<2>, // 1. dropout 50% of input
//!     Linear<5, 3>,    // 2. pass into a linear layer
//!     LayerNorm1D<3>,  // 3. normalize elements
//!     ReLU,            // 4. activate with relu
//!     Residual<(       // 5. residual connection that adds input to the result of it's sub layers
//!         Linear<3, 3>,// 5.a. Apply linear layer
//!         ReLU,        // 5.b. Apply Relu
//!     )>,              // 5.c. the input to the residual is added back in after the sub layers
//!     Linear<3, 2>,    // 6. Apply another linear layer
//! );
//! ```
//!
//! # Saving and Loading
//!
//! Call [SaveToNpz::save()] and [LoadFromNpz::load()] traits. All modules provided here implement it,
//! including tuples. These all save to/from `.npz` files, which are basically zip files with multiple `.npy`
//!  files.
//!
//! This is implemented to be fairly portable. For example you can load a simple MLP into pytorch like so:
//!
//! ```python
//! import torch
//! import numpy as np
//! state_dict = {k: torch.from_numpy(v) for k, v in np.load("dfdx-model.npz").items()}
//! mlp.load_state_dict(state_dict)
//! ```

mod activations;
mod dropout;
mod generalized_residual;
mod impl_module_for_tuples;
mod layer_norm;
mod linear;
mod module;
mod npz;
mod repeated;
mod residual;
mod split_into;

pub use activations::*;
pub use dropout::*;
pub use generalized_residual::*;
pub use impl_module_for_tuples::*;
pub use layer_norm::*;
pub use linear::*;
pub use module::*;
pub use npz::*;
pub use repeated::*;
pub use residual::*;
pub use split_into::*;

#[cfg(feature = "nightly")]
mod transformer;

#[cfg(feature = "nightly")]
pub use transformer::*;

#[cfg(feature = "nightly")]
mod flatten;

#[cfg(feature = "nightly")]
pub use flatten::*;
