//! Information about the available feature flags
//!
//! # "intel-mkl"
//!
//! Enables using the `Intel MKL` libraries (assuming you installed it already) for matrix multiplication.
//!
//! Linking is currently tested & verified on the following platforms:
//!
//! - [x] Windows
//! - [x] Linux
//! - [x] macOS
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["intel-mkl"] }
//! ```
//!
//! #### Installing Intel MKL libraries
//!
//! It's pretty easy!
//!
//! You will need to install Intel MKL on your own from
//! [this page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
//!
//! `build.rs` will fail helpfully if you don't have the correct path/environment variables.
//!
//! # "nightly"
//!
//! Enables using all features that currently require the nightly rust compiler.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["nightly"] }
//! ```

#[cfg(not(feature = "intel-mkl"))]
/// The library used for BLAS. Configure with crate features.
pub const BLAS_LIB: &str = "matrix-multiply";

#[cfg(feature = "intel-mkl")]
/// The library used for BLAS. Configure with crate features.
pub const BLAS_LIB: &str = "intel-mkl";
