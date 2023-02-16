//! High level neural network building blocks such as [modules::Linear], activations, and tuples as [Module]s.
//! Also includes `.save()` & `.load()` for all [Module]s.
//!
//! # Mutable vs Immutable forwards
//!
//! This is provided as two separate traits
//!
//! 1. [ModuleMut::forward_mut()] which receives `&mut self`.
//! 2. [Module::forward()] which receives `&self`.
//!
//! **This has nothing to do with whether gradients are being tracked or not**.
//! It only controls whether the module itself can be modified. Both OwnedTape
//! and NoneTape can still be passed to both, and all modules should conform
//! to this expected behavior.
//!
//! In general, [ModuleMut::forward_mut()] should be used during training,
//! and [Module::forward()] during evaluation/testing/inference/validation.
//!
//! Here is a list of existing modules that have different behavior in these
//! two functions:
//!
//! - [modules::BatchNorm2D]
//! - [modules::DropoutOneIn]
//! - [modules::Dropout]
//!
//! # Initializing
//!
//! Use [DeviceBuildExt] for device agnostic module creation/randomization:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! type Model = Linear<5, 2>;
//! let model = dev.build_module::<Model, f32>();
//! ```
//!
//! Here, the return type depends on the device and dtype you are using.
//!
//! For example, when using device [crate::tensor::Cpu] and `f32`, the type
//! is `Linear<5, 2, f32, Cpu>`. When using
//! a `Cuda` device and `f64`, the type is `Linear<5, 2, f64, Cuda>`.
//!
//! Alternatively, you can use [BuildModule], which requires device specific model definitions:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! use dfdx::nn::modules::Linear;
//! type Dev = Cpu;
//! let dev: Dev = Default::default();
//! let model: Linear<5, 2, f32, Dev> = BuildModule::build(&dev);
//! ```
//!
//! # Resetting parameters
//!
//! All modules implement [ResetParams], which allows you to reset a module back to a randomized
//! state:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! type Model = Linear<5, 2>;
//! let mut model = dev.build_module::<Model, f32>();
//! model.reset_params();
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
mod add_into;
mod batchnorm2d;
mod conv;
mod dropout;
mod embedding;
mod flatten;
mod generalized_residual;
mod impl_module_for_tuples;
mod layer_norm;
mod linear;
mod module;
#[cfg(feature = "numpy")]
mod npz;
#[cfg(feature = "numpy")]
mod npz_impls;
mod pool2d;
mod pool_global;
mod repeated;
mod residual;
mod split_into;
mod transformer;
mod visit_tensors;

pub use module::*;
pub use visit_tensors::*;

#[cfg(feature = "numpy")]
pub use npz::*;

pub mod modules {
    /// Structs containing initialized Tensors & impls for [super::Module]. See
    /// [super::builders] for helpful utilities in creating these
    /// in a device/dtype agnostic way.
    pub use super::activations::*;
    pub use super::add_into::AddInto;
    pub use super::batchnorm2d::BatchNorm2D;
    #[cfg(feature = "nightly")]
    pub use super::conv::Conv2D;
    pub use super::dropout::{Dropout, DropoutOneIn};
    pub use super::embedding::Embedding;
    #[cfg(feature = "nightly")]
    pub use super::flatten::Flatten2D;
    pub use super::generalized_residual::GeneralizedResidual;
    pub use super::layer_norm::LayerNorm1D;
    pub use super::linear::Linear;
    #[cfg(feature = "nightly")]
    pub use super::pool2d::{AvgPool2D, MaxPool2D, MinPool2D};
    pub use super::pool_global::{AvgPoolGlobal, MaxPoolGlobal, MinPoolGlobal};
    pub use super::repeated::Repeated;
    pub use super::residual::Residual;
    pub use super::split_into::SplitInto;
    #[cfg(feature = "nightly")]
    pub use super::transformer::*;
}

pub mod builders {
    /// Simple specification of network structure, without
    /// worrying about device or dtype.
    pub use super::activations::*;
    pub use super::add_into::AddInto;
    pub use super::batchnorm2d::builder::BatchNorm2D;
    #[cfg(feature = "nightly")]
    pub use super::conv::builder::Conv2D;
    pub use super::dropout::{Dropout, DropoutOneIn};
    pub use super::embedding::builder::Embedding;
    #[cfg(feature = "nightly")]
    pub use super::flatten::Flatten2D;
    pub use super::generalized_residual::GeneralizedResidual;
    pub use super::layer_norm::builder::LayerNorm1D;
    pub use super::linear::builder::Linear;
    #[cfg(feature = "nightly")]
    pub use super::pool2d::{AvgPool2D, MaxPool2D, MinPool2D};
    pub use super::pool_global::{AvgPoolGlobal, MaxPoolGlobal, MinPoolGlobal};
    pub use super::repeated::Repeated;
    pub use super::residual::Residual;
    pub use super::split_into::SplitInto;
    #[cfg(feature = "nightly")]
    pub use super::transformer::builder::*;
}

#[cfg(test)]
mod tests {
    use crate::{gradients::Gradients, optim::ParamUpdater, shapes::Dtype, tensor::DeviceStorage};

    #[derive(Default)]
    pub struct SimpleUpdater(pub Gradients);

    impl<D: DeviceStorage, E: Dtype> ParamUpdater<D, E> for SimpleUpdater {
        fn update_param<S: crate::shapes::Shape>(
            &mut self,
            p: &mut crate::tensor::Tensor<S, E, D>,
            unused: &mut crate::optim::UnusedTensors,
        ) -> Result<(), <D>::Err> {
            if self.0.remove(p).is_none() {
                unused.add(p);
            }
            Ok(())
        }
    }
}
