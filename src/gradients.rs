use crate::prelude::*;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct UniqueId(pub(crate) usize);

pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

impl std::ops::Deref for UniqueId {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait HasUniqueId {
    fn id(&self) -> &UniqueId;
}

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
/*

match operation {
                Operation::Unary(op) => {
                    let d_grad = self.deriv(op.parent_deriv) * self.grad(op.result_grad);
                    self.gradients[op.parent_grad.index] += &d_grad;
                }
                Operation::Binary(op) => match op.op_type {
                    OpType::MatMul { m, n, o } => {
                        let result = self
                            .grad(op.result_grad)
                            .clone()
                            .into_shape((m, o))
                            .unwrap();
                        let d0 = self
                            .deriv(op.parent_derivs[0])
                            .clone()
                            .into_shape((n, o))
                            .unwrap()
                            .reversed_axes();
                        let d1 = self
                            .deriv(op.parent_derivs[1])
                            .clone()
                            .into_shape((m, n))
                            .unwrap()
                            .reversed_axes();

                        let d_grad0 = result.dot(&d0);
                        let d_grad1 = d1.dot(&result);

                        self.gradients[op.parent_grads[0].index] += &d_grad0;
                        self.gradients[op.parent_grads[1].index] += &d_grad1;
                    }
                    OpType::Broadcast(axis, keep_axis) => {
                        let d_grad = self.deriv(op.parent_derivs[0]) * self.grad(op.result_grad);
                        self.gradients[op.parent_grads[0].index] += &d_grad;

                        let broadcasted_grad = self.grad(op.result_grad).sum_axis(axis);
                        let broadcasted_grad = if keep_axis {
                            broadcasted_grad.insert_axis(axis)
                        } else {
                            broadcasted_grad
                        };
                        let d_grad = self.deriv(op.parent_derivs[1]) * &broadcasted_grad;
                        self.gradients[op.parent_grads[1].index] += &d_grad;
                    }
                    _ => {
                        for (&deriv, grad) in op.parent_derivs.iter().zip(op.parent_grads.iter()) {
                            let d_grad = self.deriv(deriv) * self.grad(op.result_grad);
                            self.gradients[grad.index] += &d_grad;
                        }
                    }
                },
            }
*/
