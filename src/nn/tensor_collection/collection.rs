#![allow(clippy::type_complexity)]

use crate::{
    prelude::{ConstShape, Device},
    shapes::{Dtype, Shape},
    tensor::{OneFillStorage, Tensor, ZeroFillStorage},
};

use super::{ModuleField, ModuleFields, TensorField};

/// A collection of named tensors. Implementing this trait will enable anything
/// that operates on tensors, including resetting, counting number of params, updating gradients,
/// building model, and converting models between devices or dtypes.
///
/// Example implementation:
/// ```rust
/// # use dfdx::prelude::*;
/// use dfdx::nn::modules::Linear;
///
/// struct Mlp<E: Dtype, D: Device<E>> {
///     l1: Linear<10, 10, E, D>,
///     l2: Linear<10, 2, E, D>,
///     relu: ReLU,
/// }
///
/// impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for Mlp<E, D> {
///     type To<E2: Dtype, D2: Device<E2>> = Mlp<E2, D2>;
///
///     fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
///         visitor: &mut V,
///     ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
///         visitor.visit_fields(
///             (
///                 // Define name of each field and how to access it, using ModuleField for Modules,
///                 // and TensorField for Tensors.
///                 Self::module("l1", |s| &s.l1, |s| &mut s.l1),
///                 Self::module("l2", |s| &s.l2, |s| &mut s.l2),
///             ),
///             // Define how to construct the collection given its fields in the order they are given
///             // above. This conversion is done using the ModuleFields trait.
///             |(l1, l2)| Mlp { l1, l2, relu: Default::default() }
///         )
///     }
/// }
///
/// let dev = Cpu::default();
/// let model = Mlp::<f32, Cpu>::build(&dev);
/// assert_eq!(132, model.num_trainable_params());
///
/// ```
pub trait TensorCollection<E: Dtype, D: Device<E>>: Sized {
    /// Type alias that specifies the how a module's type changes when using a different dtype and/or
    /// device.
    type To<E2: Dtype, D2: Device<E2>>;

    /// Specifies how to iterate through tensors or modules containted within this module, and how
    /// to contruct this module given values for its fields. Returns `Err(_)` to indicate an error,
    /// `Ok(None)` to indicate that there is no error and a module has not been built, and
    /// `Ok(Some(_))` contains `Self::Output<E2, D2>`
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err>;

    /// Creates a [ModuleFields] that represents a field that may contain one or more tensors.
    ///
    /// See also: [ModuleField], [TensorCollection].
    fn module<F1, F2, Field>(
        name: &str,
        get_ref: F1,
        get_mut: F2,
    ) -> ModuleField<F1, F2, Self, Field>
    where
        F1: FnMut(&Self) -> &Field,
        F2: FnMut(&mut Self) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        ModuleField {
            name,
            get_ref,
            get_mut,
            m: Default::default(),
            f: Default::default(),
        }
    }

    /// Creates a [ModuleFields] that represents a tensor field.
    ///
    /// See also: [TensorField], [TensorCollection], [TensorOptions].
    fn tensor<F1, F2, S>(
        name: &str,
        get_ref: F1,
        get_mut: F2,
        options: TensorOptions<S, E, D>,
    ) -> TensorField<F1, F2, Self, S, E, D>
    where
        F1: FnMut(&Self) -> &Tensor<S, E, D>,
        F2: FnMut(&mut Self) -> &mut Tensor<S, E, D>,
        S: Shape,
    {
        TensorField {
            name,
            get_ref,
            get_mut,
            options,
            m: Default::default(),
        }
    }
}

/// An object that can visit [TensorCollection]s and [Tensor]s recursively.
pub trait ModuleVisitor<T: TensorCollection<E, D>, E: Dtype, D: Device<E>>: Sized {
    type Err;
    type E2: Dtype;
    type D2: Device<Self::E2>;

    /// Visit a [TensorCollection]. Do not use this; use visit_fields instead.
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        name: &str,
        get_refs: GetRef,
        get_muts: GetMut,
    ) -> Result<Option<Field::To<Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Field,
        GetMut: FnMut(&mut T) -> &mut Field,
        Field: TensorCollection<E, D>;

    /// Visits an actual named [Tensor]. Do not use this; use visit_fields instead.
    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        name: &str,
        get_refs: GetRef,
        get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut T) -> &mut Tensor<S, E, D>;

    /// Takes something that implements [ModuleFields] and function that takes
    /// [ModuleFields::Output] and returns an instance of T.
    fn visit_fields<M: ModuleFields<T, E, D>>(
        &mut self,
        fields: M,
        builder: impl FnOnce(M::Output<Self::E2, Self::D2>) -> T::To<Self::E2, Self::D2>,
    ) -> Result<Option<T::To<Self::E2, Self::D2>>, Self::Err>;
}

impl<S: ConstShape, E: Dtype, D: Device<E>> TensorCollection<E, D> for Tensor<S, E, D> {
    type To<E2: Dtype, D2: Device<E2>> = Tensor<S, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_tensor(
            "",
            |s| s,
            |s| s,
            TensorOptions {
                do_gradient_update: true,
                reset: |_| Ok(()),
                shape: Default::default(),
            },
        )
    }
}

/// Options to change behavior of [ModuleVisitor]
#[non_exhaustive]
pub struct TensorOptions<S: Shape, E: Dtype, D: Device<E>> {
    /// Whether the tensor should be updated with gradients
    pub do_gradient_update: bool,

    /// How to reset the tensor in the future.
    pub reset: fn(&'_ mut Tensor<S, E, D>) -> Result<(), D::Err>,

    /// The [Shape] that BuildModule uses to construct the tensor
    pub shape: S,
}

impl<S: Shape, E: Dtype, D: Device<E>> TensorOptions<S, E, D> {
    /// A tensor that should be updated with gradients & reset to 0
    pub fn reset_to_zeros() -> Self
    where
        D: ZeroFillStorage<E>,
        S: ConstShape,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_zeros(),
            shape: S::default(),
        }
    }

    /// A tensor that should be updated with gradients & reset to 1
    pub fn reset_to_ones() -> Self
    where
        D: OneFillStorage<E>,
        S: ConstShape,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_ones(),
            shape: S::default(),
        }
    }

    /// A tensor that should be updated with gradients & reset with the fn passed in
    pub fn reset_with(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self
    where
        S: ConstShape,
    {
        TensorOptions {
            do_gradient_update: true,
            reset,
            shape: S::default(),
        }
    }

    /// A tensor that should **NOT** be updated with gradients & reset with the fn passed in
    pub fn detached(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self
    where
        S: ConstShape,
    {
        TensorOptions {
            do_gradient_update: false,
            reset,
            shape: S::default(),
        }
    }
}
