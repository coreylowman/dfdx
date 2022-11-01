// Helper functions for the ResetParams derive macro
use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, FromDeriveInput, Default)]
#[darling(default, attributes(reset_params), forward_attrs(allow, doc, cfg))]
struct Opts {
    answer: Option<i32>,
    etc: Option<String>,
}

pub fn gen(ast: syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let opts = Opts::from_derive_input(&ast).expect("Wrong options");
    println!("{:?}", opts);

    // let generics = add_trait_bounds(&ast.data, ast.generics);
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    // let updates = get_updates(&ast.data);

    quote! {
        impl #impl_generics dfdx::prelude::ResetParams for #name #ty_generics #where_clause {
            fn reset_params<R: Rng>(&mut self, rng: &mut R) {
                // #updates
            }
        }
    }
}
