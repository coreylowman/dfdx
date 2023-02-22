use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
};

use super::collection::{TensorCollection, TensorOptions, TensorVisitor};

use std::{string::String, vec::Vec};

pub struct RecursiveWalker<'a, M, F> {
    pub m: M,
    pub f: &'a mut F,
    pub path: &'a mut Vec<String>,
}

pub trait VisitTensors<E: Dtype, D: DeviceStorage> {
    type Container: TensorContainer;
    type Err;

    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        opts: TensorOptions<S, E, D>,
        t: <Self::Container as TensorContainer>::WithModule<'_, Tensor<S, E, D>>,
    ) -> Result<(), Self::Err>;
}

type ContainerWithModule<'a, C, M> = <C as TensorContainer>::WithModule<'a, M>;

impl<'a, M, E: Dtype, D: DeviceStorage, F: VisitTensors<E, D>> TensorVisitor<M, E, D>
    for RecursiveWalker<'a, ContainerWithModule<'a, F::Container, M>, F>
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
            m: F::Container::get_field(&mut self.m, &mut get_refs, &mut get_muts),
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
            F::Container::get_field(&mut self.m, &mut get_refs, &mut get_muts),
        )?;
        self.path.pop();
        Ok(())
    }
}

pub trait TensorContainer: 'static {
    type WithModule<'a, Mod: 'a>
    where
        Self: 'a;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field;
}

pub enum TensorRef {}
pub enum TensorMut {}

impl TensorContainer for TensorRef {
    type WithModule<'a, Mod: 'a> = &'a Mod;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        get_ref: &mut GetRef,
        _get_mut: &mut GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_ref(*module)
    }
}

impl TensorContainer for TensorMut {
    type WithModule<'a, Mod: 'a> = &'a mut Mod;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        _get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_mut(*module)
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: TensorContainer),+> TensorContainer for ($($name,)+) {
            type WithModule<'a, Mod: 'a> = ($($name::WithModule<'a, Mod>,)+);

            fn get_field<'a, Mod, Field, GetRef, GetMut>(
                module: &'a mut Self::WithModule<'_, Mod>,
                get_ref: &mut GetRef,
                get_mut: &mut GetMut,
            ) -> Self::WithModule<'a, Field>
            where
                GetRef: FnMut(&Mod) -> &Field,
                GetMut: FnMut(&mut Mod) -> &mut Field,
            {
                ($($name::get_field(&mut module.$idx, get_ref, get_mut),)+)
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

impl<T: TensorContainer> TensorContainer for std::vec::Vec<T> {
    type WithModule<'a, Mod: 'a> = std::vec::Vec<T::WithModule<'a, Mod>>;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module
            .iter_mut()
            .map(|x| T::get_field(x, get_ref, get_mut))
            .collect()
    }
}

impl<T: TensorContainer> TensorContainer for Option<T> {
    type WithModule<'a, Mod: 'a> = Option<T::WithModule<'a, Mod>>;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        module.as_mut().map(|x| T::get_field(x, get_ref, get_mut))
    }
}
