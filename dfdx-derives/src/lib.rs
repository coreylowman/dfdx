use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Index};

macro_rules! has_attr {
    ($F:expr, $Attr:expr) => {
        $F.attrs.iter().find(|a| a.path().is_ident($Attr)).is_some()
    };
}

/// Allows you to implement [dfdx::nn_traits::Module], while automatically implementing the following:
/// 1. [dfdx::nn_traits::BuildOnDevice]
/// 2. [dfdx::nn_traits::ResetParams]
/// 3. [dfdx::nn_traits::UpdateParams]
/// 4. [dfdx::nn_traits::ZeroGrads]
/// 5. [dfdx::nn_traits::SaveSafeTensors]
/// 6. [dfdx::nn_traits::LoadSafeTensors]
///
/// If your struct contains sub module configs, then you must add the `#[module]` attribute to those items. Any field that is marked with `#[module]` will be expected to implement [dfdx::nn_traits::BuildOnDevice].
///
/// You can control the name of the built struct with the `#[built(<type name>)]` attribute on the struct.
///
/// # Using CustomModule on unit structs
///
/// Here we have a unit struct that just calls a method on Tensor in the forward:
///
/// ```ignore
/// # use dfdx::prelude::*;
/// #[derive(Default, Debug, Clone, Copy, dfdx::CustomModule)]
/// pub struct Abs;
/// impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Abs {
///     type Output = Tensor<S, E, D, T>;
///     fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
///         x.try_abs()
///     }
/// }
/// ```
///
/// # Using CustomModule on structs with non-parameter fields
///
/// ```ignore
/// # use dfdx::prelude::*;
/// #[derive(Default, Debug, Clone, Copy, dfdx::CustomModule)]
/// pub struct Reshape<S: Shape>(pub S);
///
/// impl<Src: Shape, Dst: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<Src, E, D, T>>
///     for Reshape<Dst>
/// {
///     type Output = Tensor<Dst, E, D, T>;
///     fn try_forward(&self, x: Tensor<Src, E, D, T>) -> Result<Self::Output, Error> {
///         x.try_reshape_like(&self.0)
///     }
/// }
/// ```
///
/// # Using CustomModule on structs with sub modules
///
/// Here there are a couple things to note:
/// 1. We must use the `#[built(<type name>)]` to control the name of the actual built struct
/// 2. We must use that type name when implementing `Module`
/// 3. We must annotate the sub module with `#[module]`
///
/// ```ignore
/// # use dfdx::prelude::*;
/// #[derive(Debug, Clone, dfdx::CustomModule)]
/// #[built(ResidualMatMul)]
/// pub struct ResidualMatMulConfig<I: Dim, O: Dim>(#[module] pub matmul: MatMulConfig<I, O>);
///
/// impl<X: WithEmptyTape, I: Dim, O: Dim, E: Dtype, D: Device<E>> Module<X> for ResidualMatMul<I, O, E, D>
/// where
///     MatMul<I, O, E, D>: Module<X, Output = X>,
///     X: TryAdd<X, Output = X>,
/// {
///     type Output = X;
///     fn try_forward(&self, x: X) -> Result<Self::Output, Error> {
///         self.matmul.try_forward(x.with_empty_tape())?.try_add(x)
///     }
/// }
/// ```
#[proc_macro_derive(CustomModule, attributes(module, built))]
pub fn custom_module(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let builder_name = input.ident.clone();

    let mut built_generics = input.generics.clone();

    let mut has_fields_to_build = false;

    // get the generics for the impl. `Input` must be added only to the impl_generics.
    // NOTE: without cloning, `Input` will appear in both impl & ty generics.
    let mut module_generics = built_generics.clone();
    module_generics.params.push(parse_quote!(Input));

    let (built_name, struct_def) = {
        let where_clause = built_generics.make_where_clause();
        let fields = {
            match &input.data {
                Data::Struct(ref obj) => match obj.fields {
                    Fields::Named(ref fields) => {
                        let fields = fields.named.iter().map(|f| {
                            let name = &f.ident;
                            let ty = &f.ty;
                            let vis = &f.vis;
                            if has_attr!(f, "module") {
                                has_fields_to_build = true;
                                where_clause
                                    .predicates
                                    .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                                let safetensors_serialize_attr = if cfg!(feature = "safetensors") {
                                    quote!(#[serialize])
                                } else {
                                    quote!()
                                };
                                quote_spanned!(f.span()=> #[module] #safetensors_serialize_attr #vis #name: <#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built,)
                            } else {
                                quote_spanned!(f.span()=> #vis #name: #ty,)
                            }
                        });
                        quote! { {#(#fields)*} }
                    }
                    Fields::Unnamed(ref fields) => {
                        let fields = fields.unnamed.iter().map(|f| {
                            let ty = &f.ty;
                            let vis = &f.vis;
                            if has_attr!(f, "module") {
                                has_fields_to_build = true;
                                where_clause
                                    .predicates
                                    .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                                let safetensors_serialize_attr = if cfg!(feature = "safetensors") {
                                    quote!(#[serialize])
                                } else {
                                    quote!()
                                };
                                quote_spanned!(f.span()=> #[module] #safetensors_serialize_attr #vis <#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built,)
                            } else {
                                quote_spanned!(f.span()=> #vis #ty,)
                            }
                        });
                        quote! { (#(#fields)*); }
                    }
                    Fields::Unit => quote! { ; },
                },
                Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
                Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
            }
        };

        let built_name = if has_fields_to_build {
            built_generics
                .params
                .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
            built_generics
                .params
                .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
            input
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("built"))
                .map(|attr| attr.parse_args::<syn::Ident>().unwrap())
                .unwrap_or_else(|| {
                    syn::Ident::new(&format!("Device{}", builder_name), input.span())
                })
        } else {
            builder_name.clone()
        };

        let (built_impl, _, built_where) = built_generics.split_for_impl();

        let def = if has_fields_to_build {
            let safetensors_derive = if cfg!(feature = "safetensors") {
                quote!(::dfdx::SaveSafeTensors, ::dfdx::LoadSafeTensors)
            } else {
                quote!()
            };
            quote! {
                #[derive(Clone, Debug, ::dfdx::ResetParams, ::dfdx::UpdateParams, ::dfdx::ZeroGrads, #safetensors_derive)]
                pub struct #built_name #built_impl #built_where #fields
            }
        } else {
            // there are no fields to build - we still have to derive ResetParams/UpdateParams/ZeroGrads, but since
            // there aren't any fields, they will just be passthrough impls
            let mut build_generics = built_generics.clone();
            if !has_fields_to_build {
                build_generics
                    .params
                    .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
                build_generics
                    .params
                    .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
            }
            let (build_impl, _, _) = build_generics.split_for_impl();
            let (built_impl, built_ty, built_where) = built_generics.split_for_impl();

            let safetensors_impls = if cfg!(feature = "safetensors") {
                quote! {
                    impl #built_impl ::dfdx::nn_traits::SaveSafeTensors for #builder_name #built_ty #built_where {
                        fn write_safetensors(
                            &self,
                            location: &str,
                            tensors: &mut Vec<(String, ::dfdx::safetensors::Dtype, Vec<usize>, Vec<u8>)>,
                        ) {}
                    }

                    impl #built_impl ::dfdx::nn_traits::LoadSafeTensors for #builder_name #built_ty #built_where {
                        fn read_safetensors<'a>(
                            &mut self,
                            location: &str,
                            tensors: &::dfdx::safetensors::SafeTensors<'a>,
                        ) -> Result<(), ::dfdx::safetensors::SafeTensorError> {
                            Ok(())
                        }
                    }
                }
            } else {
                quote! {}
            };

            quote! {
                #safetensors_impls

                impl #build_impl ::dfdx::nn_traits::ResetParams<Elem, Dev> for #builder_name #built_ty #built_where {
                    fn try_reset_params(&mut self) -> Result<(), ::dfdx::tensor::Error> {
                        Ok(())
                    }
                }

                impl #build_impl ::dfdx::nn_traits::UpdateParams<Elem, Dev> for #builder_name #built_ty #built_where {
                    fn try_update_params<M, Optim: ::dfdx::nn_traits::Optimizer<M, Elem, Dev>>(
                        &mut self,
                        optimizer: &mut Optim,
                        gradients: &::dfdx::tensor::Gradients<Elem, Dev>,
                        missing_tensors: &mut Vec<::dfdx::tensor::UniqueId>,
                    ) -> Result<(), ::dfdx::tensor::Error> {
                        Ok(())
                    }
                }

                impl #build_impl ::dfdx::nn_traits::ZeroGrads<Elem, Dev> for #builder_name #built_ty #built_where {
                    fn try_zero_grads(&self, grads: &mut ::dfdx::tensor::Gradients<Elem, Dev>) -> Result<(), ::dfdx::tensor::Error> {
                        Ok(())
                    }
                }
            }
        };
        (built_name, def)
    };

    let impl_build_on_device = {
        let (_, builder_ty, _) = input.generics.split_for_impl();

        let mut build_generics = built_generics.clone();
        if !has_fields_to_build {
            build_generics
                .params
                .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
            build_generics
                .params
                .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
        }
        let (build_impl, _, _) = build_generics.split_for_impl();
        let (_, built_ty, built_where) = built_generics.split_for_impl();

        match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        if has_attr!(f, "module") {
                            quote_spanned! {f.span()=> #name: self.#name.try_build_on_device(device)?, }
                        } else {
                            quote_spanned! {f.span()=> #name: self.#name, }
                        }
                    });
                    quote! {
                        impl #build_impl ::dfdx::nn_traits::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, ::dfdx::tensor::Error> {
                                let built = #built_name { #(#recurse)* };
                                Ok(built)
                            }
                        }
                    }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        if has_attr!(f, "module") {
                            quote_spanned! {f.span()=> self.#index.try_build_on_device(device)?, }
                        } else {
                            quote_spanned! {f.span()=> self.#index, }
                        }
                    });
                    quote! {
                        impl #build_impl ::dfdx::nn_traits::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, ::dfdx::tensor::Error> {
                                let built = #built_name(#(#recurse)*);
                                Ok(built)
                            }
                        }
                    }
                }
                Fields::Unit => {
                    quote! {
                        impl #build_impl ::dfdx::nn_traits::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, ::dfdx::tensor::Error> {
                                Ok(#built_name)
                            }
                        }
                    }
                }
            },
            _ => unreachable!(),
        }
    };

    proc_macro::TokenStream::from(quote! {
        #struct_def
        #impl_build_on_device
    })
}

/// Implements all of the dfdx_nn traits automatically on your type. Assumes all fields on your type
/// are modules (i.e. they also implement all the dfdx_nn traits).
///
/// [dfdx::nn_traits::Module] is implemented as calling each of the fields on the type in definition order.
///
/// # Example usage
/// Here we define a simple feedforward network with 3 layers.
/// the `#[derive(Sequential)]` means the built module will execute
/// the layers in **specification** order. So it will execute in the following order:
/// 1. linear1
/// 2. act1
/// 3. linear2
/// 4. act2
/// 5. linear3
/// ```ignore
/// # use dfdx::prelude::*;
/// #[derive(Debug, Clone, dfdx::Sequential)]
/// #[built(Mlp)]
/// struct MlpConfig {
///     // Linear with compile time input size & runtime known output size
///     linear1: LinearConfig<Const<784>, usize>,
///     act1: ReLU,
///     // Linear with runtime input & output size
///     linear2: LinearConfig<usize, usize>,
///     act2: Tanh,
///     // Linear with runtime input & compile time output size.
///     linear3: LinearConfig<usize, Const<10>>,
/// }
/// ```
#[proc_macro_derive(Sequential, attributes(built))]
pub fn sequential(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let builder_name = input.ident.clone();

    let built_name = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("built"))
        .map(|attr| attr.parse_args::<syn::Ident>().unwrap())
        .unwrap_or_else(|| syn::Ident::new(&format!("Device{}", builder_name), input.span()));
    let mut built_generics = input.generics.clone();
    built_generics
        .params
        .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
    built_generics
        .params
        .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));

    // get the generics for the impl. `Input` must be added only to the impl_generics.
    // NOTE: without cloning, `Input` will appear in both impl & ty generics.
    let mut module_generics = built_generics.clone();
    module_generics.params.push(parse_quote!(Input));

    let struct_def = {
        let where_clause = built_generics.make_where_clause();
        let fields = {
            match &input.data {
                Data::Struct(ref obj) => match obj.fields {
                    Fields::Named(ref fields) => {
                        let fields = fields.named.iter().map(|f| {
                            let name = &f.ident;
                            let ty = &f.ty;
                            let vis = &f.vis;
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                            let safetensors_serialize_attr = if cfg!(feature = "safetensors") {
                                quote!(#[serialize])
                            } else {
                                quote!()
                            };
                            quote_spanned!(f.span()=> #[module] #safetensors_serialize_attr #vis #name: <#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built,)
                        });
                        quote! { #(#fields)* }
                    }
                    Fields::Unnamed(ref fields) => {
                        let fields = fields.unnamed.iter().map(|f| {
                            let ty = &f.ty;
                            let vis = &f.vis;
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                            let safetensors_serialize_attr = if cfg!(feature = "safetensors") {
                                quote!(#[serialize])
                            } else {
                                quote!()
                            };
                            quote_spanned!(f.span()=> #[module] #safetensors_serialize_attr #vis <#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built,)
                        });
                        quote! { #(#fields)* }
                    }
                    Fields::Unit => Default::default(),
                },
                Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
                Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
            }
        };

        let (built_impl, _, built_where) = built_generics.split_for_impl();

        let safetensors_derive = if cfg!(feature = "safetensors") {
            quote!(::dfdx::SaveSafeTensors, ::dfdx::LoadSafeTensors)
        } else {
            quote!()
        };

        quote! {
            #[derive(Clone, Debug, ::dfdx::ResetParams, ::dfdx::UpdateParams, ::dfdx::ZeroGrads, #safetensors_derive)]
            pub struct #built_name #built_impl #built_where {
                #fields
            }
        }
    };

    let impl_build_on_device = {
        let (_, builder_ty, _) = input.generics.split_for_impl();
        let (built_impl, built_ty, built_where) = built_generics.split_for_impl();

        match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span()=> #name: self.#name.try_build_on_device(device)?, }
                    });
                    quote! {
                        impl #built_impl ::dfdx::nn_traits::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, ::dfdx::tensor::Error> {
                                let built = #built_name {
                                    #(#recurse)*
                                };
                                Ok(built)
                            }
                        }
                    }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        quote_spanned! {f.span()=> self.#index.try_build_on_device(device)?, }
                    });
                    quote! {
                        impl #built_impl ::dfdx::nn_traits::BuildOnDevice<Elem, Dev> for #builder_name #builder_ty #built_where {
                            type Built = #built_name #built_ty;
                            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, ::dfdx::tensor::Error> {
                                #built_name(
                                    #(#recurse)*
                                )
                            }
                        }
                    }
                }
                Fields::Unit => proc_macro2::TokenStream::new(),
            },
            _ => unreachable!(),
        }
    };

    // Get's the output type of the sequential. Also adds Module bounds to the where clause.
    let mut last_ty = quote!(Input);
    let output_ty = {
        let where_clause = module_generics.make_where_clause();
        match &input.data {
            Data::Struct(ref obj) => match obj.fields {
                Fields::Named(ref fields) => {
                    fields.named.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                        where_clause
                            .predicates
                            .push(parse_quote!(<#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built: ::dfdx::nn_traits::Module<#last_ty>));
                        last_ty = parse_quote!(<<#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built as ::dfdx::nn_traits::Module<#last_ty>>::Output);
                    });
                }
                Fields::Unnamed(ref fields) => {
                    fields.unnamed.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>));
                        where_clause
                            .predicates
                            .push(parse_quote!(<#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built: ::dfdx::nn_traits::Module<#last_ty>));
                        last_ty = parse_quote!(<<#ty as ::dfdx::nn_traits::BuildOnDevice<Elem, Dev>>::Built as ::dfdx::nn_traits::Module<#last_ty>>::Output);
                    });
                }
                Fields::Unit => {}
            },
            Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
            Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
        };
        last_ty
    };

    let impl_module = {
        let (src, src_mut) = match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let (recurse, recurse_mut) = fields
                        .named
                        .iter()
                        .map(|f| {
                            let name = &f.ident;
                            (
                                quote_spanned! {f.span()=> self.#name.try_forward(x)? },
                                quote_spanned! {f.span()=> self.#name.try_forward_mut(x)? },
                            )
                        })
                        .unzip::<_, _, Vec<_>, Vec<_>>();
                    (
                        quote! { #(let x = #recurse;)* },
                        quote! { #(let x = #recurse_mut;)* },
                    )
                }
                Fields::Unnamed(ref fields) => {
                    let (recurse, recurse_mut) = fields
                        .unnamed
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let index = Index::from(i);
                            (
                                quote_spanned! {f.span()=> self.#index.try_forward(x)? },
                                quote_spanned! {f.span()=> self.#index.try_forward_mut(x)? },
                            )
                        })
                        .unzip::<_, _, Vec<_>, Vec<_>>();
                    (
                        quote! { #(let x = #recurse;)* },
                        quote! { #(let x = #recurse_mut;)* },
                    )
                }
                Fields::Unit => (quote! { let x = x; }, quote! { let x = x; }),
            },
            _ => unreachable!(),
        };

        let (_, built_ty, _) = built_generics.split_for_impl();
        let (module_impl, _, module_where) = module_generics.split_for_impl();

        quote! {
            impl #module_impl ::dfdx::nn_traits::Module<Input> for #built_name #built_ty #module_where {
                type Output = #output_ty;
                fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
                    #src
                    Ok(x)
                }
                fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, Error> {
                    #src_mut
                    Ok(x)
                }
            }
        }
    };

    proc_macro::TokenStream::from(quote! {
        #struct_def
        #impl_build_on_device
        #impl_module
    })
}

#[proc_macro_derive(ResetParams, attributes(param, module))]
pub fn reset_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let mut custom_generics = input.generics.clone();
    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Elem"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
    }

    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Dev"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
    }

    let where_clause = input.generics.make_where_clause();
    let resets = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let resets = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    if has_attr!(f, "module") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::ResetParams<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#name.try_reset_params()?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#resets)* }
            }
            Fields::Unnamed(ref fields) => {
                let resets = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    if has_attr!(f, "module") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::ResetParams<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#index.try_reset_params()?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#resets)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("ResetParams not implemented for enums."),
        Data::Union(_) => unimplemented!("ResetParams not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics ::dfdx::nn_traits::ResetParams<Elem, Dev> for #name #ty_generics #where_clause {
            fn try_reset_params(&mut self) -> Result<(), ::dfdx::tensor::Error> {
                #resets
                Ok(())
            }
        }
    })
}

#[proc_macro_derive(UpdateParams, attributes(param, module))]
pub fn update_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let struct_name = input.ident;

    let mut custom_generics = input.generics.clone();
    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Elem"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
    }

    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Dev"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
    }

    let where_clause = input.generics.make_where_clause();
    let updates = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let updates = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    if has_attr!(f, "module") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::UpdateParams<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#name.try_update_params(optimizer, gradients, missing_tensors)?;)
                    } else if has_attr!(f, "param") {
                        quote_spanned!(f.span()=>optimizer.update_tensor(&mut self.#name, gradients, missing_tensors)?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#updates)* }
            }
            Fields::Unnamed(ref fields) => {
                let updates = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    if has_attr!(f, "module") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::UpdateParams<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#index.try_update_params(optimizer, gradients, missing_tensors)?;)
                    } else if has_attr!(f, "param") {
                        quote_spanned!(f.span()=>optimizer.update_tensor(&mut self.#index, gradients, missing_tensors)?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#updates)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("UpdateParams not implemented for enums."),
        Data::Union(_) => unimplemented!("UpdateParams not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics ::dfdx::nn_traits::UpdateParams<Elem, Dev> for #struct_name #ty_generics #where_clause {
            fn try_update_params<_Model, Optim: ::dfdx::nn_traits::Optimizer<_Model, Elem, Dev>>(
                &mut self,
                optimizer: &mut Optim,
                gradients: &::dfdx::tensor::Gradients<Elem, Dev>,
                missing_tensors: &mut Vec<::dfdx::tensor::UniqueId>,
            ) -> Result<(), ::dfdx::tensor::Error> {
                #updates
                Ok(())
            }
        }
    })
}

#[proc_macro_derive(ZeroGrads, attributes(param, module))]
pub fn zero_grads(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let mut custom_generics = input.generics.clone();
    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Elem"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Elem: ::dfdx::prelude::Dtype));
    }

    if !custom_generics.params.iter().any(
        |param| matches!(param, syn::GenericParam::Type(type_param) if type_param.ident == "Dev"),
    ) {
        custom_generics
            .params
            .push(parse_quote!(Dev: ::dfdx::prelude::Device<Elem>));
    }

    let where_clause = input.generics.make_where_clause();
    let zero_grads = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let zero_grads = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    if has_attr!(f, "module")
                    {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::ZeroGrads<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#name.try_zero_grads(grads)?;)
                    } else if has_attr!(f, "param")
                    {
                        quote_spanned!(f.span()=>self.#name.device().try_fill_with_zeros(grads.get_or_alloc_mut(&self.#name)?)?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#zero_grads)* }
            }
            Fields::Unnamed(ref fields) => {
                let zero_grads = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    if has_attr!(f, "module")
                    {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::ZeroGrads<Elem, Dev>));
                        quote_spanned!(f.span()=>self.#index.try_zero_grads(grads)?;)
                    } else if has_attr!(f, "param")
                    {
                        quote_spanned!(f.span()=>self.#index.device().try_fill_with_zeros(grads.get_or_alloc_mut(&self.#index)?)?)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#zero_grads)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("ZeroGrads not implemented for enums."),
        Data::Union(_) => unimplemented!("ZeroGrads not implemented for unions."),
    };

    let (impl_generics, _, _) = custom_generics.split_for_impl();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        impl #impl_generics ::dfdx::nn_traits::ZeroGrads<Elem, Dev> for #name #ty_generics #where_clause {
            fn try_zero_grads(&self, grads: &mut ::dfdx::prelude::Gradients<Elem, Dev>) -> Result<(), ::dfdx::tensor::Error> {
                #zero_grads
                Ok(())
            }
        }
    })
}

#[proc_macro_derive(SaveSafeTensors, attributes(serialize))]
pub fn save_safetensors(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let where_clause = input.generics.make_where_clause();
    let save_fields = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let save_fields = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    let name_str = name.as_ref().map(|n| n.to_string());
                    if has_attr!(f, "serialize")
                    {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::SaveSafeTensors));
                        quote_spanned!(f.span()=>self.#name.write_safetensors(&format!("{location}{}", #name_str), tensors);)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#save_fields)* }
            }
            Fields::Unnamed(ref fields) => {
                let save_fields = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    if has_attr!(f, "serialize")
                    {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::SaveSafeTensors));
                        quote_spanned!(f.span()=>self.#index.write_safetensors(&format!("{location}{}", #index), tensors);)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#save_fields)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("SaveSafeTensors not implemented for enums."),
        Data::Union(_) => unimplemented!("SaveSafeTensors not implemented for unions."),
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        // note: SaveSafeTensors definition is already gated by the safetensors feature
        impl #impl_generics ::dfdx::nn_traits::SaveSafeTensors for #name #ty_generics #where_clause {
            fn write_safetensors(
                &self,
                location: &str,
                tensors: &mut Vec<(String, ::dfdx::safetensors::Dtype, Vec<usize>, Vec<u8>)>,
            ) {
                #save_fields
            }
        }
    })
}

#[proc_macro_derive(LoadSafeTensors, attributes(serialize))]
pub fn load_safetensors(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let where_clause = input.generics.make_where_clause();
    let load_fields = match &input.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let load_fields = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    let name_str = name.as_ref().map(|n| n.to_string());
                    if has_attr!(f, "serialize") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::LoadSafeTensors));
                        quote_spanned!(f.span()=>self.#name.read_safetensors(&format!("{location}{}", #name_str), tensors)?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#load_fields)* }
            }
            Fields::Unnamed(ref fields) => {
                let load_fields = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let index = Index::from(i);
                    let ty = &f.ty;
                    if has_attr!(f, "serialize") {
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: ::dfdx::nn_traits::LoadSafeTensors));
                        quote_spanned!(f.span()=>self.#index.read_safetensors(&format!("{location}{}", #index), tensors)?;)
                    } else {
                        Default::default()
                    }
                });
                quote! { #(#load_fields)* }
            }
            Fields::Unit => Default::default(),
        },
        Data::Enum(_) => unimplemented!("LoadSafeTensors not implemented for enums."),
        Data::Union(_) => unimplemented!("LoadSafeTensors not implemented for unions."),
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    proc_macro::TokenStream::from(quote! {
        // note: LoadSafeTensors definition is already gated by the safetensors feature
        impl #impl_generics ::dfdx::nn_traits::LoadSafeTensors for #name #ty_generics #where_clause {
            fn read_safetensors<'a>(
                &mut self,
                location: &str,
                tensors: &::dfdx::safetensors::SafeTensors<'a>,
            ) -> Result<(), ::dfdx::safetensors::SafeTensorError> {
                #load_fields
                Ok(())
            }
        }
    })
}

/// Generates a module containing helpful structs and implementations for a input wrapper.
///
/// ## Example
///
/// The following definition:
/// ```ignore
/// #[input_wrapper]
/// pub struct MyWrapper<A, B> {
///     pub a: A,
///     pub b: B,
/// }
/// ```
///
/// Generates the following module:
/// ```ignore
/// pub mod my_wrapper {
///     // structs for the fields
///     pub struct a;
///     pub struct b;
///     // note: if MyWrapper was a tuple-struct,
///     // the fields would be named _0, _1 and so on
///
///     // structs to help in tuple conversions (Module impls omitted)
///     pub struct FromTuple;
///     pub struct IntoTuple;
///     
///     // access for the `a` field
///     impl<M: Module<A>, A, B> Module<MyWrapper<A, B>> for On<a, M>
///     {
///         type Output = MyWrapper<<M as Module<A>>::Output, B>;
///         fn try_forward(&self, x: MyWrapper<A, B>) -> Result<Self::Output, Error> {/* (...) */}
///         fn try_forward_mut(&mut self, x: MyWrapper<A, B>) -> Result<Self::Output, Error> {/* (...) */}
///     }
///
///     // access for the `b` field
///     impl<M: Module<B>, A, B> Module<MyWrapper<A, B>> for On<b, M>
///     {
///         type Output = MyWrapper<A, <M as Module<B>>::Output>;
///         fn try_forward(&self, x: MyWrapper<A, B>) -> Result<Self::Output, Error> {/* ... */}
///         fn try_forward_mut(&mut self, x: MyWrapper<A, B>) -> Result<Self::Output, Error> {/* ... */}
///     }
/// }
/// ```
/// To better visualize the generated code and items, it's recommended to expand it with Rust-Analyzer,
/// or to generate the project's documentation.
///
/// Those helpers can then be used as modules:
/// ```ignore
/// #[derive(Default, Clone, Sequential)]
/// pub struct Arch {
///     // (...)
///
///     // assuming Input is of type (X, Y), converts the input into MyWrapper<X, Y>
///     pub input_to_wrapper: my_wrapper::FromTuple,
///     
///     // apply module T on the field `a`, while also mapping the input into:
///     // MyWrapper<<T as Module<X>>::Output, Y>
///     pub t: On<my_wrapper::a, T>,
///     
///     // converts the input into a tuple:
///     // (<T as Module<X>>::Output, Y)
///     pub input_to_tuple: split1::IntoTuple,
///     
///     // (...)
/// }
/// ```
#[proc_macro_attribute]
pub fn input_wrapper(
    _attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let wrapper = parse_macro_input!(input as DeriveInput);

    // - TODO: any bounds on the struct definition probably should be copied into other impls.
    // - NOTE: check on how to deal with the `Self` (as this won't refer to the struct on the On Module impl).

    // eg. MyWrapper
    let wrapper_ident = wrapper.ident.clone();
    let wrapper_vis = wrapper.vis.clone();
    let wrapper_lowercase = format!("{}", heck::AsSnakeCase(wrapper_ident.to_string()));
    // eg. my_wrapper
    // TODO: allow renaming
    let wrapper_lowercase_ident = syn::Ident::new(&wrapper_lowercase, wrapper_ident.span());

    // get wrapper field info
    // eg. [(pub, Some(my_field), MyFieldType, field span)]
    let mut wrapper_fields = vec![];
    match &wrapper.data {
        Data::Struct(ref obj) => match obj.fields {
            Fields::Named(ref fields) => {
                let fields = fields.named.iter().map(|f| {
                    let ty = &f.ty;
                    assert_ne!(
                        quote!(#ty).to_string(),
                        "M",
                        "A generic type named `M` is not allowed because this is used internally"
                    );
                    (&f.vis, &f.ident, &f.ty, f.span())
                });
                wrapper_fields.extend(fields)
            }
            Fields::Unnamed(ref fields) => {
                let fields = fields.unnamed.iter().map(|f: &syn::Field| {
                    let ty = &f.ty;
                    assert_ne!(
                        quote!(#ty).to_string(),
                        "M",
                        "A generic type named `M` is not allowed because this is used internally"
                    );
                    (&f.vis, &None, &f.ty, f.span())
                });
                wrapper_fields.extend(fields)
            }
            // no fields
            Fields::Unit => {}
        },
        Data::Enum(_) => unimplemented!("Input wrapper cannot be derived for enums."),
        Data::Union(_) => unimplemented!("Input wrapper cannot be derived for unions."),
    };

    // wrapper fields as structs
    let mut wrapper_field_structs_quote = vec![];
    let mut are_fields_named = false;
    for (i, (_vis, field, _ty, span)) in wrapper_fields.iter().enumerate() {
        let (doc, field) = if let Some(field) = field {
            are_fields_named = true;
            let doc = format!(
                "Indicates the [`{}::{}`] field.  \nThis field is the `{}` value (`0`-based index).",
                wrapper_ident,
                field,
                i
            );
            (doc, field.clone())
        } else {
            let doc = format!(
                "Indicates the `{}`-th value from [`{}`] (0-based index).",
                i, wrapper_ident
            );
            let field = syn::Ident::new(&format!("_{}", i), *span);
            (doc, field)
        };
        wrapper_field_structs_quote.push(quote! {
            #[doc = #doc]
            #[allow(non_camel_case_types)]
            #[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct #field;
        });
    }

    let imports = if are_fields_named {
        quote! {
            use super::#wrapper_ident;
        }
    } else {
        quote! {
            use super::#wrapper_ident;
            // TODO: import tuple stuff
            use crate::prelude;
        }
    };

    let wrapper_generics = wrapper.generics.clone();
    let wrapper_generics_params = wrapper_generics.params.iter().collect::<Vec<_>>();
    // eg. MyWrapper<A, B> -> [A, B]
    let wrapper_generics_param_idents = {
        wrapper_generics_params
            .iter()
            .map(|p| {
                use syn::GenericParam::*;
                match p {
                    Lifetime(l) => &l.lifetime.ident,
                    Type(t) => &t.ident,
                    Const(c) => &c.ident,
                }
            })
            .collect::<Vec<_>>()
    };

    // eg. MyWrapper<A, B> -> [A, B]
    let wrapper_generic_names = wrapper_generics_param_idents.iter().collect::<Vec<_>>();

    // eg. MyWrapper<A> { field1: A, field2: bool} -> [A, bool]
    let field_ty_names = wrapper_fields
        .iter()
        .map(|(_, _, ty, _)| ty)
        .collect::<Vec<_>>();

    // create structs to represent tuple conversions
    let tuple_conversion_structs = {
        let field_ty_names = field_ty_names
            .iter()
            .map(|ty| quote! {#ty}.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let wrapper_generic_names = wrapper_generic_names
            .iter()
            .map(|ident| ident.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let doc1 = format!(
            "Indicates a conversion from a ({}) tuple into a `{}<{}>`.",
            &field_ty_names, wrapper_ident, &wrapper_generic_names
        );
        let doc2 = format!(
            "Indicates a conversion from a `{}<{}>` into a ({}) tuple.",
            wrapper_ident, &wrapper_generic_names, &field_ty_names
        );
        quote! {
            #[doc = #doc1]
            #[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ::dfdx::CustomModule)]
            pub struct FromTuple;
            #[doc = #doc2]
            #[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ::dfdx::prelude::CustomModule)]
            pub struct IntoTuple;
        }
    };

    // impl From<> conversions
    let tuple_conversions = {
        let doc1 = format!("Conversion of a tuple into a [`{}`].", wrapper_ident,);
        let doc2 = format!("Conversion of a [`{}`] into a tuple.", wrapper_ident,);

        let mut field_from_tuple = vec![];
        let mut field_to_tuple = vec![];
        for (i, (_, ident, _, _span)) in wrapper_fields.iter().enumerate() {
            let i = syn::Index::from(i);
            if let Some(ident) = ident {
                field_from_tuple.push(quote! {#ident: x.#i});
                field_to_tuple.push(quote! {x.#ident});
            } else {
                field_from_tuple.push(quote! {x.#i});
                field_to_tuple.push(quote! {x.#i});
            };
        }

        let (from_tuple, to_tuple) = if are_fields_named {
            (
                quote! {
                    #wrapper_ident {
                        #(#field_from_tuple), *
                    }
                },
                quote! { (#(#field_to_tuple), *) },
            )
        } else {
            (
                quote! {
                    #wrapper_ident (
                        #(#field_from_tuple), *
                    )
                },
                quote! {(#(#field_to_tuple), *)},
            )
        };

        quote! {
            #[doc = #doc1]
            impl<#(#wrapper_generic_names), *> From<(#(#field_ty_names), *)> for #wrapper_ident<#(#wrapper_generic_names), *> {
                fn from(x: (#(#field_ty_names), *)) -> Self {
                    #from_tuple
                }
            }
            #[doc = #doc2]
            impl<#(#wrapper_generic_names), *> From<#wrapper_ident<#(#wrapper_generic_names), *>> for (#(#field_ty_names), *) {
                fn from(x: #wrapper_ident<#(#wrapper_generic_names), *>) -> Self {
                    #to_tuple
                }
            }
        }
    };

    // impl Module for conversions into and from tuples
    let module_conversions = {
        let doc1 = format!("Module to convert a tuple into a [`{}`].", wrapper_ident,);
        let doc2 = format!("Module to convert a [`{}`] into a tuple.", wrapper_ident,);
        quote! {
            #[doc = #doc1]
            impl<#(#wrapper_generic_names), *> ::dfdx::prelude::Module<(#(#field_ty_names), *)> for FromTuple {
                type Output = #wrapper_ident<#(#wrapper_generic_names), *>;
                fn try_forward(&self, x: (#(#field_ty_names), *)) -> Result<Self::Output, ::dfdx::prelude::Error> {
                    Ok(x.into())
                }
            }
            #[doc = #doc2]
            impl<#(#wrapper_generic_names), *> ::dfdx::prelude::Module<#wrapper_ident<#(#wrapper_generic_names), *>> for IntoTuple {
                type Output = (#(#field_ty_names), *);
                fn try_forward(&self, x: #wrapper_ident<#(#wrapper_generic_names), *>) -> Result<Self::Output, ::dfdx::prelude::Error> {
                    Ok(x.into())
                }
            }
        }
    };

    // assertion
    for generic_ident in wrapper_generic_names.iter() {
        let count = wrapper_fields
            .iter()
            .map(|(_, _, field_ty, _)| is_ident_container(generic_ident, field_ty))
            .filter(|contains| *contains)
            .count();
        if count > 1 {
            panic!("the generic {generic_ident} should be used in at most one field");
        }
    }

    // field access modules
    let mut field_access_modules = vec![];
    for (i, (_vis, ident, ty, span)) in wrapper_fields.iter().enumerate() {
        let (doc, on_acccess, forward) = if let Some(ident) = ident {
            let doc = format!(
                "Module that access [`{}::{}`] and then applies Module `M` on it.",
                wrapper_ident, ident
            );
            let on_access = ident.clone();
            let forward = syn::Ident::new(&format!("x{i}"), ident.span());
            (doc, on_access, forward)
        } else {
            let doc = format!(
                "Module that access the `{}`-th value from [`{}`] and then applies Module `M` on it.",
                i,
                wrapper_ident,
            );
            let on_access = syn::Ident::new(&format!("_{}", i), *span);
            let forward = syn::Ident::new(&format!("x{i}"), *span);
            (doc, on_access, forward)
        };

        let mut contains_ident = false;
        let output_generics = wrapper_generic_names.iter().map(|ty_ident| {
            //
            if is_ident_container(ty_ident, ty) {
                if contains_ident {
                    panic!(
                        "the field {ident:?} at index {i} should contain at most one generic type"
                    );
                }
                contains_ident = true;
                quote!(<M as ::dfdx::prelude::Module<#ty>>::Output)
            } else {
                quote!(#ty_ident)
            }
        });

        let mut field_extraction_idents = vec![];
        let mut field_extraction = vec![];
        let mut field_construction = vec![];
        for (i, (_, _ident, _, span)) in wrapper_fields.iter().enumerate() {
            let ii = syn::Index::from(i);
            if let Some(_ident) = _ident {
                let xident = syn::Ident::new(&format!("x{i}"), _ident.span());
                field_extraction_idents.push(xident.clone());
                field_extraction.push(quote! {let #xident = x.#_ident;});
                field_construction.push(quote! {#_ident: #xident,});
            } else {
                let xident = syn::Ident::new(&format!("x{i}"), *span);
                field_extraction_idents.push(xident.clone());
                field_extraction.push(quote! {let #xident = x.#ii;});
                field_construction.push(quote! {#xident,});
            };
        }
        let field_replacement = if are_fields_named {
            quote! {
                #wrapper_ident {
                    #(#field_construction)*
                }
            }
        } else {
            quote! {
                #wrapper_ident (
                    #(#field_construction)*
                )
            }
        };

        let field_access_module = quote! {
            #[doc = #doc]
            impl<M: ::dfdx::prelude::Module<#ty>, #(#wrapper_generic_names), *> ::dfdx::prelude::Module<#wrapper_ident<#(#wrapper_generic_names), *>> for ::dfdx::prelude::On<#on_acccess, M> {
                type Output = #wrapper_ident<#(#output_generics), *>;
                fn try_forward(&self, x: #wrapper_ident<#(#wrapper_generic_names), *>) -> Result<Self::Output, ::dfdx::prelude::Error> {
                    #(#field_extraction)*
                    let #forward = self.t.try_forward(#forward)?;
                    let x = #field_replacement;
                    Ok(x)
                }
                fn try_forward_mut(&mut self, x: #wrapper_ident<#(#wrapper_generic_names), *>) -> Result<Self::Output, ::dfdx::prelude::Error> {
                    #(#field_extraction)*
                    let #forward = self.t.try_forward_mut(#forward)?;
                    let x = #field_replacement;
                    Ok(x)
                }
            }
        };
        field_access_modules.push(field_access_module);
    }

    // all of the generated content
    let _mod = quote! {
        #wrapper_vis mod #wrapper_lowercase_ident {
            #imports

            #(#wrapper_field_structs_quote)*

            #tuple_conversion_structs

            #tuple_conversions

            #module_conversions

            #(#field_access_modules)*
        }
    };
    let output = quote!(
        #wrapper

        #[doc = "Automatically generated by `input_wrapper`. The containing items are visible on your project's documentation."]
        #_mod
    );
    proc_macro::TokenStream::from(output)
}

/// Checks whether `ty` contains any ident that matches the `ident`.
fn is_ident_container(ident: &syn::Ident, ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Array(_) => todo!("input_wrapper is_ident_container for array"),
        syn::Type::BareFn(_) => todo!("input_wrapper is_ident_container for bare fn"),
        syn::Type::Group(_) => todo!("input_wrapper is_ident_container for group"),
        syn::Type::ImplTrait(_) => todo!("input_wrapper is_ident_container for impl trait"),
        syn::Type::Infer(_) => todo!("input_wrapper is_ident_container for infer"),
        syn::Type::Macro(_) => todo!("input_wrapper is_ident_container for macro"),
        syn::Type::Never(_) => todo!("input_wrapper is_ident_container for never"),
        syn::Type::Paren(_) => todo!("input_wrapper is_ident_container for paren"),
        syn::Type::Path(ty) => {
            let mut is = false;
            if let Some(qself) = &ty.qself {
                is |= is_ident_container(ident, &qself.ty);
            }
            if let Some(segment) = &ty.path.segments.last() {
                is |= &segment.ident == ident;
            }
            is
        }
        syn::Type::Ptr(_) => todo!("input_wrapper is_ident_container for ptr"),
        syn::Type::Reference(_) => todo!("input_wrapper is_ident_container for reference"),
        syn::Type::Slice(_) => todo!("input_wrapper is_ident_container for slice"),
        syn::Type::TraitObject(_) => todo!("input_wrapper is_ident_container for trait object"),
        syn::Type::Tuple(_) => todo!("input_wrapper is_ident_container for tuple"),
        syn::Type::Verbatim(_) => todo!("input_wrapper is_ident_container for verbatim"),
        other => unimplemented!(
            "input_wrapper is_ident_container not implemented for {}",
            quote!(#other).to_string()
        ),
    }
}
