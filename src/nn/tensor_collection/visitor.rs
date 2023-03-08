use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
};

use super::collection::{ModuleVisitor, TensorCollection, TensorOptions};

use std::string::String;

/// A standard [ModuleVisitor] that executes `F` on every [Tensor] encountered.
/// `F` must implement [TensorVisitor]
#[derive(Debug)]
pub struct RecursiveWalker<'a, M, F> {
    pub m: M,
    pub f: &'a mut F,
}

/// Something that can visit [Tensor]s. Used in conjunction with [RecursiveWalker].
pub trait TensorVisitor<E: Dtype, D: DeviceStorage> {
    /// The type of tensor this struct uses. E.g. [ViewTensorMut], or [ViewTensorRef]
    type Viewer: TensorViewer;
    type Err;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: <Self::Viewer as TensorViewer>::View<'_, Tensor<S, E, D>>,
    ) -> Result<(), Self::Err>;
}

/// Something that can view [Tensor]s in different ways. For example
/// [ViewTensorRef] can view `&Tensor`, and [ViewTensorMut] can view `&mut Tensor.
pub trait TensorViewer: 'static {
    type View<'a, Mod: 'a>
    where
        Self: 'a;

    /// Given a view of a module, returns a view of one of that module's fields
    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        name: &str,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field;
}

/// A [TensorViewer] that represents a `&Tensor`
#[derive(Debug)]
pub enum ViewTensorRef {}

/// A [TensorViewer] that represents a `&mut Tensor`
#[derive(Debug)]
pub enum ViewTensorMut {}

/// A [TensorViewer] that represents a Tensor's name as a `String`
#[derive(Debug)]
pub enum ViewTensorName {}

impl<'a, M, E: Dtype, D: DeviceStorage, F: TensorVisitor<E, D>> ModuleVisitor<M, E, D>
    for RecursiveWalker<'a, <F::Viewer as TensorViewer>::View<'a, M>, F>
{
    type Err = F::Err;

    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        let mut walker = RecursiveWalker {
            m: F::Viewer::view_field(&mut self.m, name, &mut get_refs, &mut get_muts),
            f: self.f,
        };
        Field::iter_tensors(&mut walker)?;
        Ok(())
    }

    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut M) -> &mut Tensor<S, E, D>,
    {
        self.f.visit(
            opts,
            F::Viewer::view_field(&mut self.m, name, &mut get_refs, &mut get_muts),
        )?;
        Ok(())
    }
}

impl TensorViewer for ViewTensorRef {
    type View<'a, Mod: 'a> = &'a Mod;

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut &Mod,
        _name: &str,
        get_ref: &mut GetRef,
        _get_mut: &mut GetMut,
    ) -> &'a Field
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_ref(module)
    }
}

impl TensorViewer for ViewTensorMut {
    type View<'a, Mod: 'a> = &'a mut Mod;

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut &mut Mod,
        _name: &str,
        _get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> &'a mut Field
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_mut(module)
    }
}

impl TensorViewer for ViewTensorName {
    type View<'a, Mod: 'a> = String;

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut String,
        name: &str,
        _get_ref: &mut GetRef,
        _get_mut: &mut GetMut,
    ) -> String
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        if !module.is_empty() {
            std::format!("{module}.{name}")
        } else {
            name.to_string()
        }
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: TensorViewer),+> TensorViewer for ($($name,)+) {
            type View<'a, Mod: 'a> = ($($name::View<'a, Mod>,)+);

            fn view_field<'a, Mod, Field, GetRef, GetMut>(
                module: &'a mut Self::View<'_, Mod>,
                name: &str,
                get_ref: &mut GetRef,
                get_mut: &mut GetMut,
            ) -> Self::View<'a, Field>
            where
                GetRef: FnMut(&Mod) -> &Field,
                GetMut: FnMut(&mut Mod) -> &mut Field,
            {
                ($($name::view_field(&mut module.$idx, name, get_ref, get_mut),)+)
            }
        }
    }
}

tuple_impls!([M1][0]);
tuple_impls!([M1, M2] [0, 1]);
tuple_impls!([M1, M2, M3] [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5]);

impl<T: TensorViewer> TensorViewer for std::vec::Vec<T> {
    type View<'a, Mod: 'a> = std::vec::Vec<T::View<'a, Mod>>;

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        name: &str,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module
            .iter_mut()
            .map(|x| T::view_field(x, name, get_ref, get_mut))
            .collect()
    }
}

impl<T: TensorViewer> TensorViewer for Option<T> {
    type View<'a, Mod: 'a> = Option<T::View<'a, Mod>>;

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        name: &str,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module.as_mut().map(|x| T::view_field(x, name, get_ref, get_mut))
    }
}
