use crate::arrays::HasArrayType;
use crate::prelude::*;
use crate::unique_id::{HasUniqueId, UniqueId};
use std::marker::PhantomData;

/// A fake tensor that holds a [UniqueId] and a type `T` that is [HasArrayType].
/// This is created and stored in gradient operations to access gradient data
/// for a tensor that the operation doesn't have ownership of.
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
    type Dtype = T::Dtype;
    type Array = T::Array;
}

impl<T: HasDevice> HasDevice for PhantomTensor<T> {
    type Device = T::Device;
}

/// Something that can be turned into a [PhantomTensor]
pub trait IntoPhantom: HasArrayData + Sized {
    fn phantom(&self) -> PhantomTensor<Self>;
}

impl<T: Tensor> IntoPhantom for T {
    /// Copies the [UniqueId] of the [Tensor], and stores the [Tensor]s array type.
    fn phantom(&self) -> PhantomTensor<Self> {
        PhantomTensor {
            id: *self.id(),
            marker: PhantomData,
        }
    }
}
