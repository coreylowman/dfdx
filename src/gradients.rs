use crate::prelude::*;
use std::collections::HashMap;

pub trait IsNdArray {
    type ArrayType: 'static
        + Sized
        + Clone
        + ZipMapElements<Self::ArrayType>
        + MapElements
        + ZeroElements
        + CountElements
        + ReduceElements
        + FillElements;
}

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
    pub(crate) fn add_operation<F: 'static + FnOnce(&mut Gradients) -> ()>(
        &mut self,
        operation: F,
    ) {
        self.operations.insert(0, Box::new(operation));
    }

    pub fn backward<T: HasUniqueId + IsNdArray>(mut self, t: &T) -> Gradients {
        let mut gradients: Gradients = Default::default();
        gradients.mut_gradient(t).fill_with(&mut || 1.0);
        for operation in self.operations.drain(..) {
            (operation)(&mut gradients);
        }
        gradients
    }
}

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
    pub fn mut_gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> &mut T::ArrayType {
        self.gradient_by_id
            .entry(*t.id())
            .or_insert_with(|| Box::new(T::ArrayType::ZEROS))
            .downcast_mut()
            .unwrap()
    }

    pub fn ref_gradient<T: HasUniqueId + IsNdArray>(&self, t: &T) -> &T::ArrayType {
        self.gradient_by_id
            .get(t.id())
            .unwrap()
            .downcast_ref()
            .unwrap()
    }

    pub fn remove_gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> Option<T::ArrayType> {
        self.gradient_by_id
            .remove_entry(t.id())
            .map(|(_, v)| *v.downcast().expect("Unable to cast properly"))
    }
}

pub trait GradientProvider {
    fn gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> Option<T::ArrayType>;
}

pub trait CanUpdateWithGradients {
    fn update<G: GradientProvider>(&mut self, grads: &mut G);
}

impl GradientProvider for Gradients {
    fn gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> Option<T::ArrayType> {
        self.remove_gradient(t)
    }
}

#[cfg(test)]
mod tests {
    use crate::array_ops::AddElements;

    use super::*;

    struct Tensor {
        id: UniqueId,
    }

    impl HasUniqueId for Tensor {
        fn id(&self) -> &UniqueId {
            &self.id
        }
    }

    impl IsNdArray for Tensor {
        type ArrayType = [f32; 5];
    }

    #[test]
    fn test_backward() {
        let t1: Tensor = Tensor { id: UniqueId(0) };
        let _t1: Tensor = Tensor { id: UniqueId(0) };

        let mut tape = GradientTape::default();
        tape.add_operation(move |g| {
            g.mut_gradient(&_t1).add(&[1.0; 5]);
        });
        let g = tape.backward(&t1);
        assert_eq!(g.ref_gradient(&t1), &[1.0; 5]);
    }
}
