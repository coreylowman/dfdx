#![allow(clippy::type_complexity)]

use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, OneFillStorage, Tensor, ZeroFillStorage},
};

use std::{string::String, vec::Vec};

pub struct TensorOptions<S: Shape, E: Dtype, D: DeviceStorage> {
    pub update: bool,
    pub reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>,
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorOptions<S, E, D> {
    pub fn reset_to_zeros() -> Self
    where
        D: ZeroFillStorage<E>,
    {
        TensorOptions {
            update: true,
            reset: |t| t.try_fill_with_zeros(),
        }
    }
    pub fn reset_to_ones() -> Self
    where
        D: OneFillStorage<E>,
    {
        TensorOptions {
            update: true,
            reset: |t| t.try_fill_with_ones(),
        }
    }
    pub fn reset_with(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            update: true,
            reset,
        }
    }
    pub fn detached(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            update: false,
            reset,
        }
    }
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

pub trait TensorContainer {
    type WithModule<'a, Mod: 'a>
    where
        Self: 'a;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        get_ref: GetRef,
        get_mut: GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field;
}

impl TensorContainer for &'static () {
    type WithModule<'a, Mod: 'a> = &'a Mod;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        mut get_ref: GetRef,
        _get_mut: GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_ref(*module)
    }
}

impl TensorContainer for &'static mut () {
    type WithModule<'a, Mod: 'a> = &'a mut Mod;

    fn get_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::WithModule<'_, Mod>,
        _get_ref: GetRef,
        mut get_mut: GetMut,
    ) -> Self::WithModule<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
        get_mut(*module)
    }
}

pub trait TensorCollection<E: Dtype, D: DeviceStorage>: Sized {
    fn iter_tensors<V: TensorVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorCollection<E, D> for Tensor<S, E, D> {
    fn iter_tensors<V: TensorVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            |s| s,
            |s| s,
            "",
            TensorOptions {
                update: true,
                reset: |_| Ok(()),
            },
        )
    }
}

pub trait TensorVisitor<T, E: Dtype, D: DeviceStorage>: Sized {
    type Err;
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        get_refs: GetRef,
        get_muts: GetMut,
        name: &str,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&T) -> &Field,
        GetMut: FnMut(&mut T) -> &mut Field,
        Field: TensorCollection<E, D>;

    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        get_refs: GetRef,
        get_muts: GetMut,
        name: &str,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&T) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut T) -> &mut Tensor<S, E, D>;
}

pub(crate) struct RecursiveWalker<'a, M, F> {
    pub(crate) m: M,
    pub(crate) f: &'a mut F,
    pub(crate) path: &'a mut Vec<String>,
}

impl<'a, M, E: Dtype, D: DeviceStorage, F: VisitTensors<E, D>> TensorVisitor<M, E, D>
    for RecursiveWalker<'a, ContainerWithModule<'a, F::Container, M>, F>
{
    type Err = F::Err;

    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        get_refs: GetRef,
        get_muts: GetMut,
        name: &str,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.path.push(name.into());
        let mut walker = RecursiveWalker {
            m: F::Container::get_field(&mut self.m, get_refs, get_muts),
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
        get_refs: GetRef,
        get_muts: GetMut,
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
            F::Container::get_field(&mut self.m, get_refs, get_muts),
        )?;
        self.path.pop();
        Ok(())
    }
}
