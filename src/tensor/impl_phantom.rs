use crate::prelude::*;
use std::marker::PhantomData;

#[derive(Clone, Copy)]
pub struct PhantomTensor<T> {
    id: UniqueId,
    marker: PhantomData<*const T>,
}

impl<T> HasUniqueId for PhantomTensor<T> {
    fn id(&self) -> &UniqueId {
        &self.id
    }
}

impl<T: HasArrayType> HasArrayType for PhantomTensor<T> {
    type Array = T::Array;
}

pub trait IntoPhantom: HasArrayData + Sized {
    fn phantom(&self) -> PhantomTensor<Self>;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H> IntoPhantom for $typename<$($Vs, )* H> {
    fn phantom(&self) -> PhantomTensor<Self> {
        PhantomTensor { id: self.id, marker: PhantomData }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
