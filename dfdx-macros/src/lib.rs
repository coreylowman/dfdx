//! Macros for use with dfdx

// This `extern` is required for older `rustc` versions but newer `rustc`
// versions warn about the unused `extern crate`.
#[allow(unused_extern_crates)]
extern crate proc_macro;

use proc_macro2::TokenStream;
use procout::procout;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{DeriveInput, parse_macro_input};


/// Implements CanUpdateWithGradients for a Module
///
/// ```rust
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
/// ```
#[proc_macro_derive(CanUpdateWithGradients)]
pub fn derive_can_update_with_gradients(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = &ast.generics.split_for_impl();

    let updates = get_updates(&ast.data);

    let code_block = quote! {

        use dfdx::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};

        impl #impl_generics CanUpdateWithGradients for #name #ty_generics #where_clause {
            fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
                #updates
            }
        }

    };

    // TODO: remove when merging
    procout(&TokenStream::from(code_block.clone()), Some(format_ident!("{}", name)), Some("macro-out"));

    proc_macro::TokenStream::from(code_block)
}

fn get_updates(data: &syn::Data) -> TokenStream {
    // Currently only works if a struct doesn't have a mix of named and unnamed fields

    // TODO: have attributes to mark which fields to add grads to (see batchnorm).
    //  maybe one attribute `nograd` and all the fields below won't have grads?

    match *data {
        syn::Data::Struct(ref data) => {
            match data.fields {
                syn::Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span() =>
                            self.#name.update(grads, unused);
                        }
                    });
                    quote! {
                        #(#recurse)*
                    }
                },
                syn::Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = syn::Index::from(i);
                        quote_spanned! {f.span() =>
                            self.#index.update(grads, unused);
                        }
                    });
                    quote! {
                        #(#recurse)*
                    }
                },
                syn::Fields::Unit => quote! {},
            }
        },
        syn::Data::Enum(_) | syn::Data::Union(_) => unimplemented!(),
    }
}