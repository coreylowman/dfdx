#![allow(clippy::type_complexity)]

use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
};

use std::{string::String, vec::Vec};

pub struct TensorOptions<S: Shape, E: Dtype, D: DeviceStorage> {
    pub name: &'static str,
    pub update: bool,
    pub reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>,
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorOptions<S, E, D> {
    pub fn named(
        name: &'static str,
        reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>,
    ) -> Self {
        Self {
            name,
            update: true,
            reset,
        }
    }

    pub fn no_grad(
        name: &'static str,
        reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>,
    ) -> Self {
        Self {
            name,
            update: false,
            reset,
        }
    }
}

pub trait VisitTensorRef<E: Dtype, D: DeviceStorage> {
    type Err;
    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<(), Self::Err>;
}

pub trait VisitTensorMut<E: Dtype, D: DeviceStorage> {
    type Err;
    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        opts: TensorOptions<S, E, D>,
        t: &mut Tensor<S, E, D>,
    ) -> Result<(), Self::Err>;
}

pub trait VisitTensorMutRef<E: Dtype, D: DeviceStorage> {
    type Err;
    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        opts: TensorOptions<S, E, D>,
        ts: (&mut Tensor<S, E, D>, &Tensor<S, E, D>),
    ) -> Result<(), Self::Err>;
}

pub trait TensorCollection<E: Dtype, D: DeviceStorage>: Sized {
    fn iter_tensors<V: TensorVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorCollection<E, D> for Tensor<S, E, D> {
    fn iter_tensors<V: TensorVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            |s| s,
            |s| s,
            TensorOptions {
                name: "",
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

impl<'a, M, E: Dtype, D: DeviceStorage, F: VisitTensorRef<E, D>> TensorVisitor<M, E, D>
    for RecursiveWalker<'a, &'a M, F>
{
    type Err = F::Err;
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        _: GetMut,
        name: &str,
    ) -> Result<(), Self::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.path.push(name.into());
        let mut walker = RecursiveWalker {
            m: get_refs(self.m),
            f: self.f,
            path: self.path,
        };
        Field::iter_tensors(&mut walker)?;
        self.path.pop();
        Ok(())
    }
    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        _: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), F::Err>
    where
        GetRef: FnMut(&M) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut M) -> &mut Tensor<S, E, D>,
    {
        self.path.push(opts.name.into());
        self.f.visit(self.path.join("."), opts, get_refs(self.m))?;
        self.path.pop();
        Ok(())
    }
}

impl<'a, M, E: Dtype, D: DeviceStorage, F: VisitTensorMut<E, D>> TensorVisitor<M, E, D>
    for RecursiveWalker<'a, &'a mut M, F>
{
    type Err = F::Err;
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        _: GetRef,
        mut get_muts: GetMut,
        name: &str,
    ) -> Result<(), F::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.path.push(name.into());
        let mut walker = RecursiveWalker {
            m: get_muts(self.m),
            f: self.f,
            path: self.path,
        };
        Field::iter_tensors(&mut walker)?;
        self.path.pop();
        Ok(())
    }
    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        _: GetRef,
        mut get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), F::Err>
    where
        GetRef: FnMut(&M) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut M) -> &mut Tensor<S, E, D>,
    {
        self.path.push(opts.name.into());
        self.f.visit(self.path.join("."), opts, get_muts(self.m))?;
        self.path.pop();
        Ok(())
    }
}

impl<'a, M, E: Dtype, D: DeviceStorage, F: VisitTensorMutRef<E, D>> TensorVisitor<M, E, D>
    for RecursiveWalker<'a, (&'a mut M, &'a M), F>
{
    type Err = F::Err;
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        name: &str,
    ) -> Result<(), F::Err>
    where
        GetRef: FnMut(&M) -> &Field,
        GetMut: FnMut(&mut M) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.path.push(name.into());
        let mut walker = RecursiveWalker {
            m: (get_muts(self.m.0), get_refs(self.m.1)),
            f: self.f,
            path: self.path,
        };
        Field::iter_tensors(&mut walker)?;
        self.path.pop();
        Ok(())
    }
    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<(), F::Err>
    where
        GetRef: FnMut(&M) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut M) -> &mut Tensor<S, E, D>,
    {
        self.path.push(opts.name.into());
        let tensors = (get_muts(self.m.0), get_refs(self.m.1));
        self.f.visit(self.path.join("."), opts, tensors)?;
        self.path.pop();
        Ok(())
    }
}
