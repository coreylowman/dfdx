//! # Architecture Configuration vs Models
//!
//! `dfdx-nn` differentiates between the *architecture* of a model, and the constructed model (that has parameters on the device).
//! This is mainly to make specifying architecture not dependent on the dtype and device
//! that a model is stored on.
//!
//! For example, a linear model has a couple pieces:
//! 1. The architecture configuration type: [LinearConfig]
//! 2. The actual built type that contains the parameters: [Linear]
//!
//! There's a third piece for convenience: [LinearConstConfig], which let's you specify dimensions at compile time.
//!
//! For specifying architecture, you just need the dimensions for the linear, but not the device/dtype:
//! ```rust
//! # use dfdx::prelude::*;
//! # use dfdx_nn::{LinearConfig, LinearConstConfig};
//! let _: LinearConfig<usize, usize> = LinearConfig::new(3, 5);
//! let _: LinearConfig<Const<3>, usize> = LinearConfig::new(Const, 5);
//! let _: LinearConfig<usize, Const<5>> = LinearConfig::new(3, Const);
//! let _: LinearConfig<Const<3>, Const<5>> = LinearConfig::new(Const, Const);
//! let _: LinearConfig<Const<3>, Const<5>> = Default::default();
//! let _: LinearConstConfig<3, 5> = Default::default();
//! ```
//! **Note** that we don't have any idea on what device or what dtype this will be.
//!
//! When we build this configuration into a [Linear] object, it will be placed on a device and have a certain dtype.
//!
//! # Building a model from an architecture
//!
//! We will use [BuildModuleExt::build_module()], an extension trait on devices, to actually construct a model.
//!
//! ```rust
//! # use dfdx_nn::*;
//! # use dfdx::prelude::*;
//! let dev: Cpu = Default::default();
//! let arch = LinearConfig::new(Const::<3>, 5);
//! let model: Linear<Const<3>, usize, f32, Cpu> = dev.build_module::<f32>(arch);
//! ```
//!
//! Notice here we have to give both the architecture configuration and a dtype. Since we are calling this method
//! on a specific device, we also end up giving the model the device it will be located on.
//!
//! # Using a model
//!
//! There are many things you can do with models. The main action is calling [Module::forward()] and [Module::forward_mut()]
//! during inference and training.
//!
//! ```rust
//! # use dfdx_nn::*;
//! # use dfdx::prelude::*;
//! # let dev: Cpu = Default::default();
//! let arch = LinearConfig::new(Const::<3>, 5);
//! let model = dev.build_module::<f32>(arch);
//! let x: Tensor<(Const<3>,), f32, _> = dev.sample_normal();
//! let y = model.forward(x);
//! assert_eq!(y.shape(), &(5, ));
//! ```
//!
//! # Composing layers into Sequential models
//!
//! There are multiple ways of doing this.
//!
//! The recommended way is to derive Sequential because:
//! 1. You can reference fields/submodules with named items instead of indexing into tuples.
//! 2. Error messages of deeply nested models are more readable.
//!
//! Under the hood, the code generated for Sequential vs tuples are identical.
//!
//! ## Deriving [Sequential]
//!
//! See [Sequential] for more detailed information.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # use dfdx_nn::*;
//! # let dev: Cpu = Default::default();
//! #[derive(Debug, Clone, Sequential)]
//! #[built(Mlp)]
//! struct MlpConfig {
//!     // Linear with compile time input size & runtime known output size
//!     linear1: LinearConfig<Const<784>, usize>,
//!     act1: ReLU,
//!     // Linear with runtime input & output size
//!     linear2: LinearConfig<usize, usize>,
//!     act2: Tanh,
//!     // Linear with runtime input & compile time output size.
//!     linear3: LinearConfig<usize, Const<10>>,
//! }
//!
//! // fill in the dimensions for the architecture
//! let arch = MlpConfig {
//!     linear1: LinearConfig::new(Const, 256),
//!     act1: Default::default(),
//!     linear2: LinearConfig::new(256, 128),
//!     act2: Default::default(),
//!     linear3: LinearConfig::new(128, Const),
//! };
//! let mut model = dev.build_module::<f32>(arch);
//! let x: Tensor<(usize, Const<784>), f32, _> = dev.sample_uniform_like(&(100, Const));
//! let y = model.forward_mut(x);
//! assert_eq!(y.shape(), &(100, Const::<10>));
//! ```
//!
//! ## Tuples
//! The simplest is to create a tuple of layer configs, which represents sequential models.
//!
//! Here's an example of how this works:
//!
//! ```rust
//! # use dfdx::prelude::*;
//! # use dfdx_nn::*;
//! # let dev: Cpu = Default::default();
//! type Arch = (LinearConstConfig<3, 5>, ReLU, LinearConstConfig<5, 10>);
//! let mut model = dev.build_module::<f32>(Arch::default());
//! let x: Tensor<(usize, Const<3>), f32, _> = dev.sample_uniform_like(&(100, Const));
//! let y = model.forward_mut(x);
//! assert_eq!(y.shape(), &(100, Const::<10>));
//! ```
//!
//! # Optimizers and Gradients
//!
//! *See [optim] for more information*
//!
//! dfdx-nn supports a number of the standard optimizers:
//!
//! | Optimizer | dfdx | pytorch |
//! | --- | --- | --- |
//! | SGD | [optim::Sgd] | `torch.optim.SGD` |
//! | Adam | [optim::Adam] | torch.optim.Adam` |
//! | AdamW | [optim::Adam] with [optim::WeightDecay::Decoupled] | `torch.optim.AdamW` |
//! | RMSprop | [optim::RMSprop] | `torch.optim.RMSprop` |
//!
//! You can use optimizers to optimize neural networks (or even tensors!). Here's
//! a simple example of how to do this:
//! ```rust
//! # use dfdx::prelude::*;
//! # use dfdx_nn::*;
//! # let dev: Cpu = Default::default();
//! type Arch = (LinearConstConfig<3, 5>, ReLU, LinearConstConfig<5, 10>);
//! let arch = Arch::default();
//! let mut model = dev.build_module::<f32>(arch);
//! // 1. allocate gradients for the model
//! let mut grads = model.alloc_grads();
//! // 2. create our optimizer
//! let mut opt = dfdx_nn::optim::Sgd::new(&model, Default::default());
//! // 3. trace gradients through forward pass
//! let x: Tensor<Rank2<10, 3>, f32, _> = dev.sample_normal();
//! let y = model.forward_mut(x.traced(grads));
//! // 4. compute loss & run backpropagation
//! let loss = y.square().mean();
//! grads = loss.backward();
//! // 5. apply gradients
//! opt.update(&mut model, &grads);
//! ```

#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

pub mod nn;

pub use dfdx_core::*;

pub mod prelude {
    pub use crate::nn::*;
    pub use dfdx_core::prelude::*;
}

#[cfg(test)]
pub(crate) mod tests {
    pub use num_traits::{Float, NumCast, Zero};

    #[cfg(not(feature = "cuda"))]
    pub type TestDevice = super::tensor::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = super::tensor::Cuda;

    #[cfg(all(feature = "test-f64", feature = "test-f16"))]
    compile_error!("f64 and f16 cannot be tested at the same time");

    #[cfg(all(
        not(feature = "test-amp-f16"),
        not(feature = "test-f16"),
        not(feature = "test-f64")
    ))]
    pub type TestDtype = f32;

    #[cfg(feature = "test-f16")]
    pub type TestDtype = crate::dtypes::f16;

    #[cfg(feature = "test-f64")]
    pub type TestDtype = f64;

    #[cfg(feature = "test-amp-f16")]
    pub type TestDtype = crate::dtypes::AMP<crate::dtypes::f16>;

    pub trait AssertClose {
        type Elem: std::fmt::Display + std::fmt::Debug + Copy;
        const DEFAULT_TOLERANCE: Self::Elem;
        fn get_default_tol(&self) -> Self::Elem {
            Self::DEFAULT_TOLERANCE
        }
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)>;
        fn assert_close(&self, rhs: &Self, tolerance: Self::Elem)
        where
            Self: std::fmt::Debug,
        {
            if let Some((l, r)) = self.get_far_pair(rhs, tolerance) {
                panic!("lhs != rhs | {l} != {r}\n\n{self:?}\n\n{rhs:?}");
            }
        }
    }

    impl<F: Copy + std::fmt::Debug + std::fmt::Display + AssertClose> AssertClose
        for crate::dtypes::AMP<F>
    {
        type Elem = crate::dtypes::AMP<F::Elem>;
        const DEFAULT_TOLERANCE: Self::Elem = crate::dtypes::AMP(F::DEFAULT_TOLERANCE);
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            self.0
                .get_far_pair(&rhs.0, tolerance.0)
                .map(|(l, r)| (crate::dtypes::AMP(l), crate::dtypes::AMP(r)))
        }
    }

    #[cfg(feature = "f16")]
    impl AssertClose for half::f16 {
        type Elem = Self;
        const DEFAULT_TOLERANCE: Self::Elem = half::f16::from_f32_const(1e-2);
        fn get_far_pair(&self, rhs: &Self, tolerance: Self) -> Option<(Self, Self)> {
            if num_traits::Float::abs(self - rhs) > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl AssertClose for f32 {
        type Elem = f32;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl AssertClose for f64 {
        type Elem = f64;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f64) -> Option<(f64, f64)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl<T: AssertClose, const M: usize> AssertClose for [T; M] {
        type Elem = T::Elem;
        const DEFAULT_TOLERANCE: Self::Elem = T::DEFAULT_TOLERANCE;
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            for (l, r) in self.iter().zip(rhs.iter()) {
                if let Some(pair) = l.get_far_pair(r, tolerance) {
                    return Some(pair);
                }
            }
            None
        }
    }

    pub trait NdMap {
        type Elem;
        type Mapped<O>;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O>;
    }

    impl NdMap for f32 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl NdMap for f64 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl<T: NdMap, const M: usize> NdMap for [T; M] {
        type Elem = T::Elem;
        type Mapped<O> = [T::Mapped<O>; M];
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O> {
            self.map(|t| t.ndmap(f))
        }
    }

    macro_rules! assert_close_to_literal {
        ($Lhs:expr, $Rhs:expr) => {{
            let lhs = $Lhs.array();
            let rhs = $Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
            let tol = AssertClose::get_default_tol(&lhs);
            let far_pair = AssertClose::get_far_pair(&lhs, &rhs, tol);
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
        ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
            let far_pair = $Lhs.array().get_far_pair(
                &$Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap()),
                num_traits::FromPrimitive::from_f64($Tolerance).unwrap(),
            );
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
    }
    pub(crate) use assert_close_to_literal;
}
