//! Implementations of [OwnedTape], [NoneTape], and generic Nd array containers via [Gradients].
#![allow(clippy::type_complexity)]

use std::collections::{HashMap, HashSet};
use std::{boxed::Box, vec::Vec};

use super::{
    storage_traits::{AllocGrad, DeviceStorage},
    unique_id, Tensor, UniqueId,
};
use crate::shapes::{Shape, Unit};

/// A generic container for keeping gradients of tensors keyed by the
/// tensor's [UniqueId].
///
/// You can:
/// 1. Insert array values into it
/// 2. Remove entries
/// 3. Access references to arrays
/// 4. Access mutable references to arrays
#[derive(Clone, Debug)]
pub struct Gradients<E: Unit, D: DeviceStorage> {
    gradient_by_id: HashMap<UniqueId, D::Vec<E>>,
    leaf_ids: Option<HashSet<UniqueId>>,
}

impl<E: Unit, D: DeviceStorage> Gradients<E, D> {
    /// Creates a [Gradients] object without any leaf tensor ids.
    /// **This will never drop gradients for temporary tensors**.
    ///
    /// This is why this method is called `leaky`, because
    /// it will keep gradients from previous passes if it is
    /// used consecutively.
    ///
    /// **You should use [crate::nn::ZeroGrads::alloc_grads]**,
    /// which will ensure non-leaf gradients are freed after backwards.
    pub fn leaky() -> Self {
        Self {
            gradient_by_id: Default::default(),
            leaf_ids: None,
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

    /// Drops all gradients except for the ids specified in the parameter.
    pub fn retain_leafs(&mut self, ids: &[UniqueId]) {
        self.leaf_ids
            .get_or_insert_with(Default::default)
            .extend(ids);
        self.drop_non_leafs();
    }

    /// Keeps all gradients marked previously by [Gradients::retain_leafs], and drops all
    /// others.
    pub fn drop_non_leafs(&mut self) {
        if let Some(leafs) = &self.leaf_ids {
            self.gradient_by_id.retain(|k, _| leafs.contains(k));
        }
    }

    /// Returns a reference to the underlying gradient if found.
    pub(crate) fn get_ref_checked<S: Shape, T>(
        &self,
        t: &Tensor<S, E, D, T>,
    ) -> Option<&D::Vec<E>> {
        self.gradient_by_id.get(&t.id)
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

/// Contains a [Gradients] and list of backward operations.
pub struct OwnedTape<E: Unit, D: DeviceStorage> {
    /// A list of (Time, BackwardOp) pairs. The Time is used to ensure operations
    /// from merged tapes are executed in the correct order.
    pub(crate) operations: Vec<(UniqueId, BackwardOp<E, D, D::Err>)>,
    pub(crate) gradients: Gradients<E, D>,
}

impl<E: Unit, D: DeviceStorage> Default for OwnedTape<E, D> {
    fn default() -> Self {
        Self {
            operations: Default::default(),
            gradients: Gradients::leaky(),
        }
    }
}

impl<E: Unit, D: DeviceStorage> std::fmt::Debug for OwnedTape<E, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedTape")
            .field("num_operations", &self.operations.len())
            .field("gradients", &self.gradients)
            .finish()
    }
}

impl<E: Unit, D: DeviceStorage> OwnedTape<E, D> {
    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub(crate) fn execute(mut self) -> Result<Gradients<E, D>, D::Err> {
        // We must ensure that the operations are sorted in execution time order.
        // Otherwise an backward operation may not be executed in the right order
        // if multiple tapes were merged together.
        self.operations.sort_by_key(|(k, _)| *k);
        for (_, operation) in self.operations.drain(..).rev() {
            (operation)(&mut self.gradients)?;
        }
        Ok(self.gradients)
    }
}

type BackwardOp<E, D, Err> = Box<dyn FnOnce(&mut Gradients<E, D>) -> Result<(), Err>>;

/// Contains nothing. When [Tape::add_backward_op] is called, this struct does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

/// Something that can track backward operations.
pub trait Tape<E: Unit, D: DeviceStorage>: Default + Merge<Self> + Merge<NoneTape> {
    /// Whether this object is currently tracking gradients. This is known at compile time.
    const OWNS_TAPE: bool;
    fn add_backward_op<F>(&mut self, operation: F)
    where
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>;
    fn try_alloc_grad<S: Shape>(&mut self, t: &Tensor<S, E, D>) -> Result<(), D::Err>;
}

impl<E: Unit, D: DeviceStorage> Tape<E, D> for OwnedTape<E, D> {
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F>(&mut self, operation: F)
    where
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>,
    {
        self.operations.push((unique_id(), Box::new(operation)));
    }
    fn try_alloc_grad<S: Shape>(&mut self, t: &Tensor<S, E, D>) -> Result<(), D::Err> {
        self.gradients.try_alloc_for(t)
    }
}

impl<E: Unit, D: DeviceStorage> Tape<E, D> for NoneTape {
    const OWNS_TAPE: bool = false;
    fn add_backward_op<F>(&mut self, _: F)
    where
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>,
    {
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
        self.gradients
            .gradient_by_id
            .extend(other.gradients.gradient_by_id.drain());
        if let Some(leafs) = other.gradients.leaf_ids {
            self.gradients
                .leaf_ids
                .get_or_insert_with(Default::default)
                .extend(leafs);
        }
        self.operations.append(&mut other.operations);
        self
    }
}
