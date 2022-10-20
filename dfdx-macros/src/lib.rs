//! Macros for use with dfdx

// This `extern` is required for older `rustc` versions but newer `rustc`
// versions warn about the unused `extern crate`.
#[allow(unused_extern_crates)]
extern crate proc_macro;

use proc_macro::TokenStream;

/// Implements CanUpdateWithGradients for a Module
///
/// ```ignore
/// use dfdx::prelude::*;
/// use dfdx_macros::CanUpdateWithGradients;
///
/// #[derive(CanUpdateWithGradients)]
/// pub struct Linear<const I: usize, const O: usize> {
///     // Transposed weight matrix, shape (O, I)
///     pub weight: Tensor2D<O, I>,
///
///     // Bias vector, shape (O, )
///     pub bias: Tensor1D<O>,
/// }
#[proc_macro_derive(CanUpdateWithGradients)]
pub fn derive_can_update_with_gradients(_input: TokenStream) -> TokenStream {
    unimplemented!()
}
