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
//! - [modules::BatchNorm1D]
//! - [modules::BatchNorm2D]
//! - [modules::DropoutOneIn]
//! - [modules::Dropout]
//!
//! # Fallible forwards
//!
//! You can also get a result from Module by using [ModuleMut::try_forward_mut],
//! and [Module::try_forward].
//!
//! Similar to fallible tensor_ops, the main purpose of this is to handle out of memory
//! errors at the device level.
//!
//! # Initializing
//!
//! Use [DeviceBuildExt] for device agnostic module creation/randomization:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! use dfdx::nn::builders::{Linear, DeviceBuildExt};
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
//! use dfdx::nn::modules::{Linear, BuildModule};
//! type Dev = Cpu;
//! let dev: Dev = Default::default();
//! let model: Linear<5, 2, f32, Dev> = BuildModule::build(&dev);
//! ```
//!
//! # Allocating & zeroing gradients
//!
//! Use [ZeroGrads::alloc_grads()] and [ZeroGrads::zero_grads()] to reduce allocations,
//! and enable gradient accumulation!
//! This is the equivalent of pytorch's `Optimizer.zero_grad`
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! # type Model = Linear<5, 2>;
//! use dfdx::nn::ZeroGrads;
//! let model = dev.build_module::<Model, f32>();
//! let mut grads: Gradients<f32, _> = model.alloc_grads();
//! model.zero_grads(&mut grads);
//! ```
//!
//! # Exponential Moving Average (EMA)
//!
//! All models implement [ModelEMA::ema()] to keep track of an exponential moving average
//! of an entire model.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! # type Model = Linear<5, 2>;
//! use dfdx::nn::ModelEMA;
//! let model = dev.build_module::<Model, f32>();
//! let mut ema_model = dev.build_module::<Model, f32>();
//! ema_model.ema(&model, 0.001);
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
//! # numpy
//!
//! Enable with the `"numpy"` feature.
//!
//! Call [SaveToNpz::save()] and [LoadFromNpz::load()] methods. All modules provided here implement it,
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
//!
//! # safetensors
//!
//! Enable with the `"safetensors"` feature.
//!
//! The feature `safetensors` allows to do the same with
//! [https://github.com/huggingface/safetensors]().
//!
//! Call [SaveToSafetensors::save_safetensors()] and [LoadFromSafetensors::load_safetensors()] funcs.
//! All modules provided here implement it, including tuples.
//!
//! These all save to/from `.safetensors` files, which are flat layout with JSON
//! header, allowing for super fast loads (with memory mapping).
//!
//! This is implemented to be fairly portable. For example you can use
//! [https://github.com/huggingface/transformers]()
//!
//! ```python
//! from transformers import pipeline
//!
//! pipe = pipeline(model="gpt2")
//! pipe.save_pretrained("my_local", safe_serialization=True)
//! # This created `my_local/model.safetensors` file which can now be used.
//! ```

mod build_module;
mod num_params;
mod reset_params;
pub mod tensor_collection;
mod to_device;
mod zero_grads;

mod module;

mod activations;
mod add_into;
mod batchnorm1d;
mod batchnorm2d;
mod bias2d;
mod conv;
mod dropout;
mod ema;
mod embedding;
mod flatten;
mod generalized_residual;
mod impl_module_for_tuples;
mod layer_norm;
mod linear;
#[cfg(feature = "numpy")]
mod npz;
mod pool2d;
mod pool_global;
mod repeated;
mod residual;
#[cfg(feature = "safetensors")]
mod safetensors;
mod split_into;
mod transformer;
mod unbiased_linear;

pub use module::{
    BuildModule, BuildOnDevice, DeviceBuildExt, Module, ModuleMut, NonMutableModule,
    ZeroSizedModule,
};

pub use tensor_collection::*;

#[cfg(feature = "safetensors")]
pub use self::safetensors::{LoadFromSafetensors, SaveToSafetensors};
pub use ema::ModelEMA;
#[cfg(feature = "numpy")]
pub use npz::{LoadFromNpz, SaveToNpz};
pub use num_params::NumParams;
pub use reset_params::ResetParams;
pub use to_device::ToDevice;
pub use zero_grads::ZeroGrads;

pub mod modules {
    //! Structs containing initialized Tensors & impls for [super::Module]. See
    //! [super::builders] for helpful utilities in creating these
    //! in a device/dtype agnostic way.
    pub use super::activations::*;
    pub use super::add_into::AddInto;
    pub use super::batchnorm1d::BatchNorm1D;
    pub use super::batchnorm2d::BatchNorm2D;
    pub use super::bias2d::Bias2D;
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
    pub use super::transformer::{
        MultiHeadAttention, Transformer, TransformerDecoder, TransformerDecoderBlock,
        TransformerEncoder, TransformerEncoderBlock,
    };
    pub use super::unbiased_linear::UnbiasedLinear;
    pub use super::*;
}

pub mod builders {
    //! Simple specification of network structure, without
    //! worrying about device or dtype.
    pub use super::activations::*;
    pub use super::add_into::AddInto;
    pub use super::batchnorm1d::builder::BatchNorm1D;
    pub use super::batchnorm2d::builder::BatchNorm2D;
    pub use super::bias2d::builder::Bias2D;
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
    pub use super::transformer::builder::{
        MultiHeadAttention, Transformer, TransformerDecoder, TransformerDecoderBlock,
        TransformerEncoder, TransformerEncoderBlock,
    };
    pub use super::unbiased_linear::builder::UnbiasedLinear;
    pub use super::*;
}
