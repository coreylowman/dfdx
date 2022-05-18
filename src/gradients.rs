use crate::prelude::*;
use std::collections::HashMap;

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

    pub fn backward<T: HasUniqueId + HasArrayType>(mut self, t: &T) -> Gradients
    where
        Cpu: FillElements<T::Array>,
    {
        let mut gradients: Gradients = Default::default();
        Cpu::fill(gradients.mut_gradient(t), &mut |f| *f = 1.0);
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
    pub fn insert<T: HasUniqueId + HasArrayType>(&mut self, t: &T, data: Box<T::Array>) {
        self.gradient_by_id.insert(*t.id(), data);
    }

    pub fn mut_gradient<T: HasUniqueId + HasArrayType>(&mut self, t: &T) -> &mut T::Array {
        self.gradient_by_id
            .entry(*t.id())
            .or_insert_with(|| Cpu::zeros::<T::Array>())
            .as_mut()
            .downcast_mut()
            .unwrap()
    }

    pub fn ref_gradient<T: HasUniqueId + HasArrayType>(&self, t: &T) -> &T::Array {
        self.gradient_by_id
            .get(t.id())
            .unwrap()
            .as_ref()
            .downcast_ref()
            .unwrap()
    }

    pub fn remove_gradient<T: HasUniqueId + HasArrayType>(
        &mut self,
        t: &T,
    ) -> Option<Box<T::Array>> {
        self.gradient_by_id
            .remove_entry(t.id())
            .map(|(_, v)| v.downcast().expect("Unable to cast properly"))
    }
}

pub trait GradientProvider {
    fn gradient<T: HasUniqueId + HasArrayType>(&mut self, t: &T) -> Option<Box<T::Array>>
    where
        Cpu: Device<T::Array>;
}

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

    #[test]
    fn test_backward() {
        let t1: Tensor = Tensor { id: UniqueId(0) };
        let _t1: Tensor = Tensor { id: UniqueId(0) };

        let mut tape = GradientTape::default();
        tape.add_operation(move |g| {
            Cpu::zip_map_assign(g.mut_gradient(&_t1), &[1.0; 5], |l, r| *l += r);
        });
        let g = tape.backward(&t1);
        assert_eq!(g.ref_gradient(&t1), &[2.0; 5]);
    }
}
