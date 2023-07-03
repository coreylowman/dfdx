use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Index};

#[proc_macro_derive(Sequential)]
pub fn sequential(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse the input tokens into a syntax tree.
    let input = parse_macro_input!(input as DeriveInput);

    let builder_name = input.ident.clone();
    let built_name = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("built"))
        .map(|attr| attr.parse_args::<syn::Ident>().unwrap())
        .unwrap_or_else(|| syn::Ident::new(&format!("{}Built", builder_name), input.span()));

    let mut built_generics = input.generics.clone();
    built_generics.params.push(parse_quote!(E: Dtype));
    built_generics.params.push(parse_quote!(D: Device<E>));

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
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: BuildOnDevice<D, E>));
                            quote_spanned!(f.span()=> #name: <#ty as BuildOnDevice<D, E>>::Built,)
                        });
                        quote! {
                            #(#fields)*
                        }
                    }
                    Fields::Unnamed(ref fields) => {
                        let fields = fields.unnamed.iter().map(|f| {
                            let ty = &f.ty;
                            where_clause
                                .predicates
                                .push(parse_quote!(#ty: BuildOnDevice<D, E>));
                            quote_spanned!(f.span()=> <#ty as BuildOnDevice<D, E>>::Built,)
                        });
                        quote! {
                            #(#fields)*
                        }
                    }
                    Fields::Unit => Default::default(),
                },
                Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
                Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
            }
        };

        let (built_impl, _, built_where) = built_generics.split_for_impl();

        quote! {
            struct #built_name #built_impl #built_where {
                #fields
            }
        }
    };

    let impl_build_on_device = {
        let (_, builder_ty, _) = input.generics.split_for_impl();
        let (built_impl, built_ty, built_where) = built_generics.split_for_impl();
        quote! {
            impl #built_impl BuildOnDevice<D, E> for #builder_name #builder_ty #built_where {
                type Built = #built_name #built_ty;
            }
        }
    };

    // Get's the output type of the sequential. Also adds Module bounds to the where clause.
    let (output_ty, output_err) = {
        let where_clause = module_generics.make_where_clause();
        let mut last_ty = quote!(Input);
        let mut last_err = quote!(<Input as HasErr>::Error);
        match &input.data {
            Data::Struct(ref obj) => match obj.fields {
                Fields::Named(ref fields) => {
                    fields.named.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: Module<#last_ty>));
                        last_ty = parse_quote!(<#ty as Module<#last_ty>>::Output);
                        last_err = parse_quote!(<#ty as Module<#last_ty>>::Error);
                    });
                }
                Fields::Unnamed(ref fields) => {
                    fields.unnamed.iter().for_each(|f| {
                        let ty = &f.ty;
                        where_clause
                            .predicates
                            .push(parse_quote!(#ty: Module<#last_ty>));
                        last_ty = parse_quote!(<#ty as Module<#last_ty>>::Output);
                        last_err = parse_quote!(<#ty as Module<#last_ty>>::Error);
                    });
                }
                Fields::Unit => {}
            },
            Data::Enum(_) => unimplemented!("Sequential cannot be derived for enums."),
            Data::Union(_) => unimplemented!("Sequential cannot be derived for unions."),
        };
        (last_ty, last_err)
    };

    let impl_tensor_collection = {
        let mut to_generics = input.generics.clone();
        to_generics.params.push(parse_quote!(E2));
        to_generics.params.push(parse_quote!(D2));
        let (_, to_ty, _) = to_generics.split_for_impl();
        let (built_impl, built_ty, built_where) = built_generics.split_for_impl();
        quote! {
            impl #built_impl TensorCollection<E, D> for #built_name #built_ty #built_where {
                type To<E2: Dtype, D2: Device<E2>> = #built_name #to_ty;
                fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
                    visitor: &mut V
                ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
                    todo!()
                }
            }
        }
    };

    let impl_module = {
        let src = match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span()=> self.#name.try_forward(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        quote_spanned! {f.span()=> self.#index.try_forward(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unit => quote! { let x = x; },
            },
            _ => unreachable!(),
        };

        let (module_impl, module_ty, module_where) = module_generics.split_for_impl();

        quote! {
            impl #module_impl Module<Input> for #built_name #module_ty #module_where {
                type Output = #output_ty;
                type Error = #output_err;
                fn try_forward(&self, x: Input) -> Result<Self::Output, Self::Error> {
                    #src
                    Ok(x)
                }
            }
        }
    };

    let impl_module_mut = {
        let src = match input.data {
            Data::Struct(ref data) => match data.fields {
                Fields::Named(ref fields) => {
                    let recurse = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        quote_spanned! {f.span()=> self.#name.try_forward_mut(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unnamed(ref fields) => {
                    let recurse = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let index = Index::from(i);
                        quote_spanned! {f.span()=> self.#index.try_forward_mut(x)? }
                    });
                    quote! { #(let x = #recurse;)* }
                }
                Fields::Unit => quote! { let x = x; },
            },
            _ => unreachable!(),
        };

        let (module_impl, module_ty, module_where) = module_generics.split_for_impl();

        quote! {
            impl #module_impl ModuleMut<Input> for #built_name #module_ty #module_where {
                type Output = #output_ty;
                type Error = #output_err;
                fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, Self::Error> {
                    #src
                    Ok(x)
                }
            }
        }
    };

    proc_macro::TokenStream::from(quote! {
        #struct_def
        #impl_build_on_device
        #impl_tensor_collection
    })
    // #impl_module
    // #impl_module_mut
}
