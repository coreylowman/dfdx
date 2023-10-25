//! Information about the available feature flags.
//!
//! Default features:
//! - "std"
//! - "fast-alloc"
//! - "cpu-par-matmul"
//!
//! # Quick start
//!
//! Cuda (with cudnn):
//! ```toml
//! dfdx = { version = "...", default-features = false, features = ["std", "fast-alloc", "cuda", "cudnn"]}
//! ```
//!
//! Cpu:
//! ```toml
//! dfdx = { version = "...", default-features = false, features = ["std", "fast-alloc", "cpu"]}
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
//! # "cudnn"
//!
//! This requires the "cuda" feature, and enables some cudnn optimizations.
//! **This is purely for performance**, no other methods/structs are introduced
//! with this feature.
//!
//! Example:
//! ```toml
//! dfdx = { version = "...", features = ["cudnn"] }
//! ```
//!
//! # "cpu"
//!
//! Used to enable using `gemm` for matmul operations.
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
