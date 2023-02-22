use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
};

use super::collection::{ModuleVisitor, TensorCollection, TensorOptions};

use std::{string::String, vec::Vec};

/// A standard [ModuleVisitor] that executes `F` on every [Tensor] encountered.
/// `F` must implement [TensorVisitor]
pub struct RecursiveWalker<'a, M, F> {
    pub m: M,
    pub f: &'a mut F,
    pub path: &'a mut Vec<String>,
}

/// Something that can visit [Tensor]s. Used in conjunction with [RecursiveWalker].
pub trait TensorVisitor<E: Dtype, D: DeviceStorage> {
    /// The type of tensor this struct uses. E.g. [TensorMut], or [TensorRef]
    type Viewer: TensorViewer;
    type Err;

    fn visit<S: Shape>(
        &mut self,
        full_path: String,
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

    /// Return the view of the tensor
    fn view<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field;
}

/// A [TensorViewer] that represents a `&Tensor`
pub enum ViewTensorRef {}

/// A [TensorViewer] that represents a `&mut Tensor`
pub enum ViewTensorMut {}

impl<'a, M, E: Dtype, D: DeviceStorage, F: TensorVisitor<E, D>> ModuleVisitor<M, E, D>
    for RecursiveWalker<'a, <F::Viewer as TensorViewer>::View<'a, M>, F>
{
    type Err = F::Err;

    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        name: &str,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.path.push(name.into());
        let mut walker = RecursiveWalker {
            m: F::Viewer::view(&mut self.m, &mut get_refs, &mut get_muts),
            f: self.f,
            path: self.path,
        };
        Field::iter_tensors(&mut walker)?;
        std::mem::drop(walker);
        self.path.pop();
        Ok(())
    }

    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        name: &str,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut M) -> &mut Tensor<S, E, D>,
    {
        self.path.push(name.into());
        self.f.visit(
            self.path.join("."),
            opts,
            F::Viewer::view(&mut self.m, &mut get_refs, &mut get_muts),
        )?;
        self.path.pop();
        Ok(())
    }
}

impl TensorViewer for ViewTensorRef {
    type View<'a, Mod: 'a> = &'a Mod;

    fn view<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        get_ref: &mut GetRef,
        _get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_ref(module)
    }
}

impl TensorViewer for ViewTensorMut {
    type View<'a, Mod: 'a> = &'a mut Mod;

    fn view<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        _get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_mut(module)
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: TensorViewer),+> TensorViewer for ($($name,)+) {
            type View<'a, Mod: 'a> = ($($name::View<'a, Mod>,)+);

            fn view<'a, Mod, Field, GetRef, GetMut>(
                module: &'a mut Self::View<'_, Mod>,
                get_ref: &mut GetRef,
                get_mut: &mut GetMut,
            ) -> Self::View<'a, Field>
            where
                GetRef: FnMut(&Mod) -> &Field,
                GetMut: FnMut(&mut Mod) -> &mut Field,
            {
                ($($name::view(&mut module.$idx, get_ref, get_mut),)+)
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

    fn view<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module
            .iter_mut()
            .map(|x| T::view(x, get_ref, get_mut))
            .collect()
    }
}

impl<T: TensorViewer> TensorViewer for Option<T> {
    type View<'a, Mod: 'a> = Option<T::View<'a, Mod>>;

    fn view<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module.as_mut().map(|x| T::view(x, get_ref, get_mut))
    }
}
