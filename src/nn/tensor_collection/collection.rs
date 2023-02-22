use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, OneFillStorage, Tensor, ZeroFillStorage},
};

/// A collection of named tensors. Implementing this trait will enable anything
/// that operates
pub trait TensorCollection<E: Dtype, D: DeviceStorage>: Sized {
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err>;
}

/// An object that can visit [TensorCollection]s and [Tensor]s recursively.
pub trait ModuleVisitor<T, E: Dtype, D: DeviceStorage>: Sized {
    type Err;

    /// Visit a [TensorCollection]
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

    /// Visits an actual named [Tensor]
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

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorCollection<E, D> for Tensor<S, E, D> {
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            |s| s,
            |s| s,
            "",
            TensorOptions {
                do_gradient_update: true,
                reset: |_| Ok(()),
            },
        )
    }
}

/// Options to change behavior of [TensorVisitor]
pub struct TensorOptions<S: Shape, E: Dtype, D: DeviceStorage> {
    /// Whether the tensor should be updated with gradients
    pub do_gradient_update: bool,
    pub reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>,
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TensorOptions<S, E, D> {
    pub fn reset_to_zeros() -> Self
    where
        D: ZeroFillStorage<E>,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_zeros(),
        }
    }
    pub fn reset_to_ones() -> Self
    where
        D: OneFillStorage<E>,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_ones(),
        }
    }
    pub fn reset_with(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            do_gradient_update: true,
            reset,
        }
    }
    pub fn detached(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            do_gradient_update: false,
            reset,
        }
    }
}
