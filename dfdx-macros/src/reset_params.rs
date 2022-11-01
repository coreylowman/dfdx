// Helper functions for the ResetParams derive macro
use darling::{FromDeriveInput, FromMeta, Result, Error};
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;



#[derive(Debug)]
enum Initializers {
    Zeros,
    Ones,
    Normal,
    // TODO: add initializers that take parameters, such as RandomUniform(min, max)
    //  should those initializers be added to the main crate?
    //  how can they be imported in both the main and macro crate? maybe they should live in a
    //  third crate?
}

impl Default for Initializers {
    fn default() -> Self {
        Initializers::Normal
    }
}

impl FromMeta for Initializers {
    fn from_string(value: &str) -> Result<Self> {
        match value {
            "zeros" => Ok(Initializers::Zeros),
            "ones" => Ok(Initializers::Ones),
            "normal" => Ok(Initializers::Normal),
            _ => Err(Error::unknown_value(value)),
        }
    }
}

#[derive(Debug, FromDeriveInput, Default)]
#[darling(default, attributes(reset_params), forward_attrs(allow, doc, cfg))]
struct Opts {
    initializer: Option<Initializers>,
}

pub fn gen(ast: syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let opts = Opts::from_derive_input(&ast).expect("Error in parsing attribute 'initializer'");

    let generics = add_trait_bounds(&ast.data, ast.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let init_code = get_initializer_code(&ast.data, &opts.initializer);
    let fields_code = get_fields_code(&ast.data, &opts.initializer);

    quote! {
        impl #impl_generics dfdx::prelude::ResetParams for #name #ty_generics #where_clause {
            fn reset_params<R: Rng>(&mut self, rng: &mut R) {
                #init_code
                #fields_code
            }
        }
    }
}


// Add a bound `T: ResetParams` to every type parameter T
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
                    .push(syn::parse_quote!(dfdx::prelude::ResetParams));
            }
        }
    }
    generics
}


// Generates `self.f.randomize(rng, &dist);` for each named field f
// or `self.x.reset_params(rng);` for each unnamed field x
fn get_fields_code(data: &syn::Data, init: &Option<Initializers>) -> TokenStream {
    match *data {
        syn::Data::Struct(ref data) => match data.fields {
            syn::Fields::Named(ref fields) => {
                let recurse = fields.named.iter().map(|f| {
                    let field_gen = get_named_field_fn(f, init);
                    quote_spanned! {f.span() => #field_gen }
                });
                quote! {
                    #(#recurse)*
                }
            }
            syn::Fields::Unnamed(ref fields) => {
                let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = syn::Index::from(i);
                    quote_spanned! {f.span() =>
                        self.#index.reset_params(rng);
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


fn get_named_field_fn(f: &syn::Field, init: &Option<Initializers>) -> TokenStream {
    let name = &f.ident;
    match init {
        Some(Initializers::Zeros) => {
            quote! { Cpu::fill(self.#name.mut_data(), &mut |v| *v = 0.0); }
        },
        Some(Initializers::Ones) => {
            quote! { Cpu::fill(self.#name.mut_data(), &mut |v| *v = 1.0); }
        },
        Some(Initializers::Normal) | None => {
            quote! { self.#name.randomize(rng, &dist); }
        },
    }
}


fn get_initializer_code(data: &syn::Data, init: &Option<Initializers>) -> TokenStream {
    let add_initializer_code = match *data {
        syn::Data::Struct(ref data) => matches!(data.fields, syn::Fields::Named(_)),
        _ => false,
    };

    if add_initializer_code {
        match init {
            Some(Initializers::Zeros) | Some(Initializers::Ones) => { quote! {} },
            Some(Initializers::Normal) | None => {
                quote! { let dist = StandardNormal; }
            }
        }
    } else {
        quote! {}
    }
}