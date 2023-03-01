//! Implementations of [GradientTape] and generic Nd array containers via [Gradients].
#![allow(clippy::type_complexity)]

use std::collections::HashMap;
use std::{boxed::Box, vec::Vec};

use crate::shapes::{Shape, Unit};
use crate::tensor::{
    storage_traits::{AllocGrad, DeviceStorage},
    Tensor,
};
use crate::unique_id::{unique_id, UniqueId};

/// A generic container for keeping variable sized arrays associated with a [UniqueId].
///
/// You can:
/// 1. Insert array values into it
/// 2. Remove entries
/// 3. Access references to arrays
/// 4. Access mutable references to arrays
#[derive(Debug)]
pub struct Gradients<E: Unit, D: DeviceStorage> {
    gradient_by_id: HashMap<UniqueId, D::Vec<E>>,
}

impl<E: Unit, D: DeviceStorage> Default for Gradients<E, D> {
    fn default() -> Self {
        Self {
            gradient_by_id: Default::default(),
        }
    }
}

impl<E: Unit, D: DeviceStorage> Gradients<E, D> {
    /// Retrieves mutable gradient for `t`, allocating one if it isn't present.
    pub(crate) fn get_or_alloc_mut<S: Shape>(
        &mut self,
        t: &Tensor<S, E, D>,
    ) -> Result<&mut D::Vec<E>, D::Err> {
        self.try_alloc_for(t)?;
        Ok(self.get_mut(t))
    }

    /// Inserts a gradient for `t`
    pub(crate) fn try_alloc_for<S: Shape>(&mut self, t: &Tensor<S, E, D>) -> Result<(), D::Err> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.gradient_by_id.entry(t.id) {
            e.insert(t.try_alloc_grad()?);
        }
        Ok(())
    }

    /// Removes and returns the data associated with `t.id()`.
    ///
    /// **Panics** if data associated with `t` is not found. This indicates an unrecoverable bug.
    pub(crate) fn remove<S: Shape, T>(&mut self, t: &Tensor<S, E, D, T>) -> Option<D::Vec<E>> {
        self.gradient_by_id.remove_entry(&t.id).map(|(_, v)| v)
    }

    /// Returns a mutable reference to the data associated with `t`.
    ///
    /// **Panics** if data associated with `t` is not found. This indicates an unrecoverable bug.
    pub(crate) fn get_mut<S: Shape, T>(&mut self, t: &Tensor<S, E, D, T>) -> &mut D::Vec<E> {
        self.gradient_by_id.get_mut(&t.id).unwrap()
    }

    /// Returns a mutable reference to the data associated with `t`.
    ///
    /// **Panics** if data associated with `t` is not found. This indicates an unrecoverable bug.
    pub(crate) fn get_ref<S: Shape, T>(&mut self, t: &Tensor<S, E, D, T>) -> &D::Vec<E> {
        self.gradient_by_id.get(&t.id).unwrap()
    }

    /// Clones the gradient and transforms it into a tensor.
    ///
    /// # Panics
    /// If no data is associated with `t` yet, this will panic due to an unwrap()
    /// on a .get() to the underlying hashmap.
    pub fn get<S: Shape, T>(&self, t: &Tensor<S, E, D, T>) -> Tensor<S, E, D> {
        let buf = self.gradient_by_id.get(&t.id).unwrap().clone();
        Tensor {
            id: unique_id(),
            data: std::sync::Arc::new(buf),
            shape: t.shape,
            strides: t.strides,
            device: t.device.clone(),
            tape: Default::default(),
        }
    }

    /// Borrows a pair of a gradients `(&mut L, &R)`.
    /// `l` is the gradient to update, and `r` is the gradient to backprop.
    ///
    /// **Panics** if `l` and `r` have the same id.
    pub(crate) fn mut_and_ref<L: Shape, R: Shape>(
        &mut self,
        l: &Tensor<L, E, D>,
        r: &Tensor<R, E, D>,
    ) -> (&mut D::Vec<E>, &D::Vec<E>) {
        assert_ne!(l.id, r.id);
        let l_ptr = self.get_mut(l) as *mut _;
        let r_ptr = self.get_ref(r) as *const _;
        let l_ref = unsafe { &mut *l_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l_ref, r_ref)
    }

    /// Borrows a triplet of gradients `(&mut L1, &mut L2, &R)`.
    pub(crate) fn muts_and_ref<L1: Shape, L2: Shape, R: Shape>(
        &mut self,
        l1: &Tensor<L1, E, D>,
        l2: &Tensor<L2, E, D>,
        r: &Tensor<R, E, D>,
    ) -> (&mut D::Vec<E>, &mut D::Vec<E>, &D::Vec<E>) {
        assert_ne!(l1.id, l2.id);
        assert_ne!(l1.id, r.id);
        assert_ne!(l2.id, r.id);
        let l1_ptr = self.get_mut(l1) as *mut _;
        let l2_ptr = self.get_mut(l2) as *mut _;
        let r_ptr = self.get_ref(r) as *const _;
        let l1_ref = unsafe { &mut *l1_ptr };
        let l2_ref = unsafe { &mut *l2_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l1_ref, l2_ref, r_ref)
    }

    #[inline]
    pub(crate) fn many_and_ref<L: Shape, R: Shape>(
        &mut self,
        ls: &Vec<Tensor<L, E, D>>,
        r: &Tensor<R, E, D>,
    ) -> (Vec<&mut D::Vec<E>>, &D::Vec<E>) {
        for i in 0..ls.len() {
            assert_ne!(ls[i].id, r.id);
            for j in (i + 1)..ls.len() {
                assert_ne!(ls[i].id, ls[j].id);
            }
        }
        let l_refs: Vec<&mut D::Vec<E>> = ls
            .iter()
            .map(|l| {
                let l_ptr = self.get_mut(l) as *mut D::Vec<E>;
                unsafe { &mut *l_ptr }
            })
            .collect();
        let r_ptr = self.get_ref(r) as *const _;
        let r_ref = unsafe { &*r_ptr };
        (l_refs, r_ref)
    }
}

/// Records gradient computations to execute later.
///
/// The only two things you can do with this are:
/// 1. Adding an operation (an operation is a FnOnce that acts on &mut [Gradients])
/// 2. Executing all the operations to produce [Gradients]
///
/// The reason for this design, which forces users to specify gradient computations, as opposed to having
/// a fixed set of *kinds* of computations are these:
/// 1. Different tensor sizes. The tensors size information would have to be stored inside the operation somehow.
///     Instead, the operation themselves must query with a sized tensor, so sizes are still known at compile time instead of dynamically.
/// 2. Slightly different operations. It'd have to support broadcasting operations, etc which can get needlessly complex.
/// 3. Optimizations are harder. With operations having control over everything, they can be optimized by hand separately.
///
/// An example for how these two are used is the following from the negate operation (ie. multiply all values by -1).
///
/// ```ignore
/// tape.add_backward_op(move |grads| {
///     let (t_grad, result_grad) = grads.mut_and_ref(&t, &_result);
///     // addmul_assign is equivalent to: t_grad += t.data() * result_grad;
///     T::Device::addmul(t_grad, t.data(), result_grad);
/// });
/// ```
///
/// This is implementing the chain rule, which is normally defined as `gradient(t) += deriv * gradient(result)` with
/// the following optimizations:
/// 1. instead of allocating new data for the derivative (which is just -1 everywhere), we can reuse the `t` tensor since the negate
///     function owns it.
/// 2. We can combine computing the derivative and multiplying by the `gradient(result)` by just setting `t` to `-gradient(result)`
///
/// This would not be possible if these chain rule operations were inside of GradientTape!
#[allow(clippy::type_complexity)]
pub struct GradientTape<E: Unit, D: DeviceStorage> {
    operations: Vec<Box<dyn FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>>>,
    gradients: Gradients<E, D>,
}

impl<E: Unit, D: DeviceStorage> Default for GradientTape<E, D> {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            gradients: Default::default(),
        }
    }
}

impl<E: Unit, D: DeviceStorage> std::fmt::Debug for GradientTape<E, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl<E: Unit, D: DeviceStorage> GradientTape<E, D> {
    /// Add an operation to be executed later. Implementation is all left to the caller,
    /// but the operation should likely call [Gradients::ref_gradient] and [Gradients::mut_gradient].
    ///
    /// # Arguments
    /// * `operation` - A FnOnce that acts on [Gradients].
    ///
    /// See src/tensor_ops for implementation examples.
    pub(crate) fn add_backward_op<
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>,
    >(
        &mut self,
        operation: F,
    ) {
        self.operations.push(Box::new(operation));
    }

    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub(crate) fn execute(mut self) -> Result<Gradients<E, D>, D::Err> {
        for operation in self.operations.drain(..).rev() {
            (operation)(&mut self.gradients)?;
        }
        Ok(self.gradients)
    }

    /// Moves all the operations from `other` into self. Leaves `other` empty.
    pub(crate) fn append(&mut self, other: &mut Self) {
        self.gradients
            .gradient_by_id
            .extend(other.gradients.gradient_by_id.drain());
        self.operations.append(&mut other.operations);
    }
}

/// Contains a boxed [GradientTape]. When [Tape::add_backward_op] is called,
/// this function passes the operation directly to [GradientTape].
#[derive(Debug, Default)]
pub struct OwnedTape<E: Unit, D: DeviceStorage>(pub(crate) Box<GradientTape<E, D>>);

/// Contains nothing. When [Tape::add_backward_op] is called, this struct does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

/// Something that can add a gradient operation to [GradientTape].
pub trait Tape<E: Unit, D: DeviceStorage>: Default + Merge<Self> + Merge<NoneTape> {
    /// Whether this object currently owns the [GradientTape]. This is known at compile time.
    const OWNS_TAPE: bool;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>>(
        &mut self,
        operation: F,
    );
    fn try_alloc_grad<S: Shape>(&mut self, t: &Tensor<S, E, D>) -> Result<(), D::Err>;
}

impl<E: Unit, D: DeviceStorage> Tape<E, D> for OwnedTape<E, D> {
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>>(
        &mut self,
        operation: F,
    ) {
        self.0.add_backward_op(operation)
    }
    fn try_alloc_grad<S: Shape>(&mut self, t: &Tensor<S, E, D>) -> Result<(), D::Err> {
        self.0.gradients.try_alloc_for(t)
    }
}

impl<E: Unit, D: DeviceStorage> Tape<E, D> for NoneTape {
    const OWNS_TAPE: bool = false;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>>(
        &mut self,
        _: F,
    ) {
    }
    fn try_alloc_grad<S: Shape>(&mut self, _: &Tensor<S, E, D>) -> Result<(), D::Err> {
        Ok(())
    }
}

/// Combine two things
pub trait Merge<T: ?Sized> {
    /// Merges `T` into `self`
    fn merge(self, other: T) -> Self;
}

impl Merge<NoneTape> for NoneTape {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<E: Unit, D: DeviceStorage> Merge<NoneTape> for OwnedTape<E, D> {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<E: Unit, D: DeviceStorage> Merge<OwnedTape<E, D>> for OwnedTape<E, D> {
    fn merge(mut self, mut other: Self) -> Self {
        self.0.append(other.0.as_mut());
        self
    }
}
