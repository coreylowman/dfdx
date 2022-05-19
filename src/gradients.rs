use crate::prelude::*;
use std::collections::HashMap;

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
/// tape_holder.add_operation(move |tape| {
///     T::Device::zip_map_assign(t.mut_data(), tape.ref_gradient(&_result), |l, r| *l = -r);
///     T::Device::add_assign(tape.mut_gradient(&t), t.data());
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
pub struct GradientTape {
    operations: Vec<Box<dyn FnOnce(&mut Gradients) -> ()>>,
}

impl std::fmt::Debug for GradientTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

impl GradientTape {
    /// Add an operation to be executed later. Implementation is all left to the caller,
    /// but the operation should likely call [Gradients::ref_gradient] and [Gradients::mut_gradient].
    ///
    /// NOTE: This adds the operation to the beginning of the list, so operations are executed
    /// in reverse order that they are added.
    ///
    /// # Arguments
    /// * `operation` - A FnOnce that acts on [Gradients].
    ///
    /// See src/tensor_ops for implementation examples.
    pub(crate) fn add_operation<F: 'static + FnOnce(&mut Gradients) -> ()>(
        &mut self,
        operation: F,
    ) {
        self.operations.insert(0, Box::new(operation));
    }

    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub fn execute(mut self) -> Gradients {
        let mut gradients: Gradients = Default::default();
        for operation in self.operations.drain(..) {
            (operation)(&mut gradients);
        }
        gradients
    }
}

/// A generic container for keeping variable sized arrays associated with a [UniqueId].
///
/// You can:
/// 1. Insert array values into it
/// 2. Remove entries
/// 3. Access references to arrays
/// 4. Access mutable references to arrays
///
/// This structure is similar to a HashMap, where all the methods require a key
/// implementing [UniqueId] and [HasArrayType].
///
/// Under the hood, it actually is a HashMap, and stores values as Box<dyn Any>. The
/// important part of key's implementing [HasArrayType] is that the associated type
/// of that trait is used to downcast the box to the expected value.
#[derive(Debug)]
pub struct Gradients {
    gradient_by_id: HashMap<UniqueId, Box<dyn std::any::Any>>,
}

impl Default for Gradients {
    fn default() -> Self {
        Self {
            gradient_by_id: HashMap::new(),
        }
    }
}

impl Gradients {
    /// Insert's `data` associated with `t.id()`.
    ///
    /// Example usage:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// gradients.insert(&t, Box::new([-4.0, 5.0, -6.0]));
    /// assert_eq!(gradients.ref_gradient(&t), &[-4.0, 5.0, -6.0]);
    /// ```
    pub fn insert<T: HasUniqueId + HasArrayType>(&mut self, t: &T, data: Box<T::Array>) {
        self.gradient_by_id.insert(*t.id(), data);
    }

    /// Removes and returns the data associated with `t.id()`.
    ///
    /// Example usage:
    /// ```
    /// # use dfdx::prelude::*;
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// gradients.insert(&t, Box::new([-4.0, 5.0, -6.0]));
    /// assert_eq!(gradients.remove(&t).unwrap().as_ref(), &[-4.0, 5.0, -6.0]);
    /// ```
    pub fn remove<T: HasUniqueId + HasArrayType>(&mut self, t: &T) -> Option<Box<T::Array>> {
        self.gradient_by_id
            .remove_entry(t.id())
            .map(|(_, v)| v.downcast().expect("Unable to cast properly"))
    }

    /// Returns a mutable reference to the data associated with `t`.
    ///
    /// If no data is associated with `t`, then [AllocateZeros::zeros] is called
    /// to allocate the data.
    ///
    /// Example usage:
    /// ```
    /// # use dfdx::prelude::*;
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// let g: &mut [f32; 3] = gradients.mut_gradient(&t);
    /// assert_eq!(g, &mut [0.0, 0.0, 0.0]);
    /// g[0] = 1.0;
    /// assert_eq!(gradients.ref_gradient(&t), &[1.0, 0.0, 0.0]);
    /// ```
    pub fn mut_gradient<T: HasUniqueId + HasArrayType + HasDevice>(
        &mut self,
        t: &T,
    ) -> &mut T::Array {
        self.gradient_by_id
            .entry(*t.id())
            .or_insert_with(|| T::Device::zeros::<T::Array>())
            .as_mut()
            .downcast_mut()
            .unwrap()
    }

    /// Returns a reference to the data associated with `t`.
    ///
    /// # Panics
    ///
    /// If no data is associated with `t` yet, this will panic due to an unwrap()
    /// on a .get() to the underlying hashmap.
    ///
    /// # Example usage:
    /// ```
    /// # use dfdx::prelude::*;
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// gradients.mut_gradient(&t);
    /// assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 0.0]);
    /// ```
    pub fn ref_gradient<T: HasUniqueId + HasArrayType>(&self, t: &T) -> &T::Array {
        self.gradient_by_id
            .get(t.id())
            .unwrap()
            .as_ref()
            .downcast_ref()
            .unwrap()
    }
}

/// Represents something that can return a gradient for a given key.
///
/// This is very similar to what [Gradients] does, however the intention
/// is that any this object be passed to [CanUpdateWithGradients].
///
/// [Gradients] does **not** implement this, so you *have* to go through
/// an optimizer to update a [CanUpdateWithGradients]. Although it very easily
/// could.
///
/// See [Sgd] and [Adam] for examples on implementing this.
pub trait GradientProvider {
    /// Retrieves the data associated with `p` if there is any.
    /// This can modify `self`, for instance if velocities are calculated
    /// based on the associated data!
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType + HasDevice;
}

/// Represents something that can be updated with [GradientProvider].
///
/// Most implementations of this trait will have sub structs that also
/// implement [CanUpdateWithGradients].
///
/// For example the [Linear] model just calls update on its weight & bias:
/// ```ignore
/// impl<const I: usize, const O: usize> CanUpdateWithGradients for Linear<I, O> {
///     fn update<G: GradientProvider>(&mut self, grads: &mut G) {
///         self.weight.update(grads);
///         self.bias.update(grads);
///     }
/// }
/// ```
pub trait CanUpdateWithGradients {
    fn update<G: GradientProvider>(&mut self, grads: &mut G);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tensor {
        id: UniqueId,
    }

    impl HasUniqueId for Tensor {
        fn id(&self) -> &UniqueId {
            &self.id
        }
    }

    impl HasArrayType for Tensor {
        type Array = [f32; 5];
    }

    impl HasDevice for Tensor {
        type Device = Cpu;
    }

    #[test]
    fn test_backward() {
        let t1: Tensor = Tensor { id: UniqueId(0) };
        let _t1: Tensor = Tensor { id: UniqueId(0) };

        let mut tape = GradientTape::default();
        tape.add_operation(move |g| {
            Cpu::zip_map_assign(g.mut_gradient(&_t1), &[1.0; 5], |l, r| *l += r);
        });
        let g = tape.execute();
        assert_eq!(g.ref_gradient(&t1), &[1.0; 5]);
    }
}
