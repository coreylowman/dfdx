use super::visitors::TensorContainer;

impl TensorContainer for &'static () {
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

impl TensorContainer for &'static mut () {
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

tuple_impls!([M1] [0]);
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
        module.iter_mut().map(|x| T::get_field(x, get_ref, get_mut)).collect()
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
