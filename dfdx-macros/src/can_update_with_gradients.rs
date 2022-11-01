// Helper functions for the CanUpdateWithGradients derive macro
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;

pub fn gen(ast: syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let generics = add_trait_bounds(&ast.data, ast.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let updates = get_updates(&ast.data);

    quote! {
        impl #impl_generics dfdx::gradients::CanUpdateWithGradients for #name #ty_generics #where_clause {
            fn update<G: dfdx::gradients::GradientProvider>(&mut self, grads: &mut G, unused: &mut dfdx::gradients::UnusedTensors) {
                #updates
            }
        }
    }
}

// Add a bound `T: CanUpdateWithGradients` to every type parameter T
// if the struct contains unnamed fields.
fn add_trait_bounds(data: &syn::Data, mut generics: syn::Generics) -> syn::Generics {
    let add_trait_bounds = match *data {
        syn::Data::Struct(ref data) => matches!(data.fields, syn::Fields::Unnamed(_)),
        _ => false,
    };

    if add_trait_bounds {
        for param in &mut generics.params {
            if let syn::GenericParam::Type(ref mut type_param) = *param {
                type_param
                    .bounds
                    .push(syn::parse_quote!(dfdx::gradients::CanUpdateWithGradients));
            }
        }
    }
    generics
}

// Generates the `self.f.update(grads, unused)` for each field
fn get_updates(data: &syn::Data) -> TokenStream {
    // TODO: have attributes to mark which fields to add grads to (see batchnorm).
    //  maybe one attribute `nograd` and all the fields below won't have grads?

    match *data {
        syn::Data::Struct(ref data) => match data.fields {
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
            }
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
            }
            syn::Fields::Unit => quote! {},
        },
        syn::Data::Enum(_) | syn::Data::Union(_) => quote!(),
    }
}
