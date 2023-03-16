//! Information about the available feature flags.
//!
//! Default features:
//! - "std"
//! - "fast-alloc"
//! - "cpu-par-matmul"
//!
//! # Quick start
//!
//! Cuda:
//! ```toml
//! dfdx = { version = "...", default-features = False, features = ["std", "cuda"]}
//! ```
//!
//! Cpu:
//! ```toml
//! dfdx = { version = "...", default-features = False, features = ["std", "cpu-par-matmul"]}
//! ```
//!
//! # "std"
//!
//! **Enabled by default**
//!
//! Enables usage of the standard library. Otherwise [no_std_compat](https://crates.io/crates/no-std-compat)
//! is used.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", default-features = false }
//! ```
//!
//! Note that allocations are necessary, so the no_std_compat dependency looks like:
//! ```toml
//! no-std-compat = { version = "0.4.1", features = [ "alloc", "compat_hash" ] }
//! ```
//!
//! # "fast-alloc"
//!
//! **Enabled by default**
//!
//! Turns off fallible allocations for Cpu, which is substantially faster.
//!
//! # "no-std"
//!
//! Used to enable "no-std-compat" and turn on `![no_std]`.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", default-features = false, features = ["no-std"] }
//! ```
//!
//! # "cuda"
//!
//! Enables the `Cuda` device and other goodies. Must have the cuda toolkit and
//! `nvcc` installed on your system.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["cuda"] }
//! ```
//!
//! # "cpu-seq-matmul"
//!
//! Used to enable using `matrixmultiply` for matmul operations.
//!
//! # "cpu-par-matmul"
//!
//! Used to enable the threading feature of `matrixmultiply`. This makes matmuls
//! substantially faster!
//!
//! # "cpu-mkl-matmul"
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
//! dfdx = { version = "...", features = ["cpu-mkl-matmul"] }
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
//! # "numpy"
//!
//! Enables saving and loading arrays to .npy files, and saving and loading nn to .npz files.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["numpy"] }
//! ```
//!
//! # "safetensors"
//!
//! Enables saving and loading tensors/nn to .safetensors files.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["safetensors"] }
//! ```
//!
//! # "nightly"
//!
//! Enables using all features that currently require the nightly rust compiler.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["nightly"] }
//! ```
