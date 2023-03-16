use std::vec::Vec;

use crate::{
    prelude::Device,
    shapes::{Dtype, Shape},
    tensor::Tensor,
};

use super::*;

impl<'a, T: TensorCollection<E, D>, E: Dtype, D: Device<E>, F: TensorVisitor<E, D>>
    ModuleVisitor<T, E, D> for RecursiveWalker<'a, <F::Viewer as TensorViewer>::View<'a, T>, F>
{
    type Err = F::Err;
    type E2 = F::E2;
    type D2 = F::D2;

    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
    ) -> Result<Option<Field::To<Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Field,
        GetMut: FnMut(&mut T) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        let mut walker = RecursiveWalker {
            m: F::Viewer::view_field(&mut self.m, name, &mut get_refs, &mut get_muts),
            f: self.f,
        };
        Field::iter_tensors(&mut walker)
    }

    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut T) -> &mut Tensor<S, E, D>,
    {
        self.f.visit(
            opts,
            F::Viewer::view_field(&mut self.m, name, &mut get_refs, &mut get_muts),
        )
    }

    fn visit_fields<M: ModuleFields<T, E, D>>(
        &mut self,
        fields: M,
        builder: impl FnOnce(M::Output<Self::E2, Self::D2>) -> T::To<Self::E2, Self::D2>,
    ) -> Result<Option<T::To<Self::E2, Self::D2>>, Self::Err> {
        let options = fields.visit_fields(self)?;
        Ok(M::handle_options(options).map(builder))
    }
}

impl TensorViewer for () {
    type View<'a, Mod: 'a> = ();

    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        _module: &mut (),
        _name: &str,
        _get_ref: &mut GetRef,
        _get_mut: &mut GetMut,
    ) where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field,
    {
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

    fn view_field<Mod, Field, GetRef, GetMut>(
        module: &mut String,
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

impl<T: TensorViewer> TensorViewer for Vec<T> {
    type View<'a, Mod: 'a> = Vec<T::View<'a, Mod>>;

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
        module
            .as_mut()
            .map(|x| T::view_field(x, name, get_ref, get_mut))
    }
}

impl<'a, F1, F2, E: Dtype, D: Device<E>, Mod: TensorCollection<E, D>, Field> ModuleFields<Mod, E, D>
    for ModuleField<'a, F1, F2, Mod, Field>
where
    F1: FnMut(&Mod) -> &Field,
    F2: FnMut(&mut Mod) -> &mut Field,
    Field: TensorCollection<E, D>,
{
    type Options<E2: Dtype, D2: Device<E2>> = Option<Field::To<E2, D2>>;
    type Output<E2: Dtype, D2: Device<E2>> = Field::To<E2, D2>;

    fn visit_fields<V: ModuleVisitor<Mod, E, D>>(
        self,
        visitor: &mut V,
    ) -> Result<Self::Options<V::E2, V::D2>, V::Err> {
        visitor.visit_module(self.name, self.get_ref, self.get_mut)
    }

    fn handle_options<E2: Dtype, D2: Device<E2>>(
        options: Self::Options<E2, D2>,
    ) -> Option<Self::Output<E2, D2>> {
        options
    }
}

impl<'a, F1, F2, S: Shape, E: Dtype, D: Device<E>, Mod: TensorCollection<E, D>>
    ModuleFields<Mod, E, D> for TensorField<'a, F1, F2, Mod, S, E, D>
where
    F1: FnMut(&Mod) -> &Tensor<S, E, D>,
    F2: FnMut(&mut Mod) -> &mut Tensor<S, E, D>,
{
    type Options<E2: Dtype, D2: Device<E2>> = Option<Tensor<S, E2, D2>>;
    type Output<E2: Dtype, D2: Device<E2>> = Tensor<S, E2, D2>;

    fn visit_fields<V: ModuleVisitor<Mod, E, D>>(
        self,
        visitor: &mut V,
    ) -> Result<Self::Options<V::E2, V::D2>, V::Err> {
        visitor.visit_tensor(self.name, self.get_ref, self.get_mut, self.options)
    }

    fn handle_options<E2: Dtype, D2: Device<E2>>(
        options: Self::Options<E2, D2>,
    ) -> Option<Self::Output<E2, D2>> {
        options
    }
}

impl<T: ModuleFields<Mod, E, D>, Mod: TensorCollection<E, D>, E: Dtype, D: Device<E>>
    ModuleFields<Mod, E, D> for Vec<T>
{
    type Options<E2: Dtype, D2: Device<E2>> = Vec<T::Options<E2, D2>>;
    type Output<E2: Dtype, D2: Device<E2>> = Vec<T::Output<E2, D2>>;

    fn visit_fields<V: ModuleVisitor<Mod, E, D>>(
        self,
        module: &mut V,
    ) -> Result<Self::Options<V::E2, V::D2>, V::Err> {
        let mut out = Vec::with_capacity(self.len());

        for x in self {
            out.push(x.visit_fields(module)?);
        }

        Ok(out)
    }

    fn handle_options<E2: Dtype, D2: Device<E2>>(
        options: Self::Options<E2, D2>,
    ) -> Option<Self::Output<E2, D2>> {
        let mut out = Vec::with_capacity(options.len());

        for x in options {
            out.push(T::handle_options(x)?);
        }

        Some(out)
    }
}

impl<Mod: TensorCollection<E, D>, E: Dtype, D: Device<E>> ModuleFields<Mod, E, D> for () {
    type Options<E2: Dtype, D2: Device<E2>> = ();
    type Output<E2: Dtype, D2: Device<E2>> = ();

    fn visit_fields<V: ModuleVisitor<Mod, E, D>>(self, _module: &mut V) -> Result<(), V::Err> {
        Ok(())
    }

    fn handle_options<E2: Dtype, D2: Device<E2>>(_options: ()) -> Option<()> {
        Some(())
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

        impl<$($name: ModuleFields<Mod, E, D>),+, Mod: TensorCollection<E, D>, E: Dtype, D: Device<E>>
            ModuleFields<Mod, E, D> for ($($name,)+)
        {
            type Options<E2: Dtype, D2: Device<E2>> = ($($name::Options<E2, D2>,)+);
            type Output<E2: Dtype, D2: Device<E2>> = ($($name::Output<E2, D2>,)+);

            fn visit_fields<V: ModuleVisitor<Mod, E, D>>(
                self,
                visitor: &mut V,
            ) -> Result<Self::Options<V::E2, V::D2>, V::Err> {
                Ok(($(self.$idx.visit_fields(visitor)?,)+))
            }

            fn handle_options<E2: Dtype, D2: Device<E2>>(
                options: Self::Options<E2, D2>,
            ) -> Option<Self::Output<E2, D2>> {
                Some(($($name::handle_options(options.$idx)?,)+))
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
