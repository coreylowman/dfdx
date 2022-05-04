use crate::prelude::*;
use std::collections::HashMap;

pub trait HasUniqueId {
    fn id(&self) -> usize;
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
        gradients.mut_gradient(t).map_assign_elems(|v| *v = 1.0);
        for operation in self.operations.drain(..) {
            (operation)(&mut gradients);
        }
        gradients
    }
}

pub struct Gradients {
    gradient_by_id: HashMap<usize, Box<dyn std::any::Any>>,
}

impl Default for Gradients {
    fn default() -> Self {
        Self {
            gradient_by_id: HashMap::new(),
        }
    }
}

impl Gradients {
    pub fn gradient<T: HasUniqueId + IsNdArray>(&self, t: &T) -> &T::ArrayType {
        self.gradient_by_id
            .get(&t.id())
            .unwrap()
            .downcast_ref()
            .unwrap()
    }

    pub fn mut_gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> &mut T::ArrayType {
        self.gradient_by_id
            .entry(t.id())
            .or_insert_with(|| Box::new(T::ArrayType::ZEROS))
            .downcast_mut()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::array_ops::AddElements;

    use super::*;

    #[derive(Default)]
    struct Tensor {
        id: usize,
    }

    impl HasUniqueId for Tensor {
        fn id(&self) -> usize {
            self.id
        }
    }

    impl IsNdArray for Tensor {
        type ArrayType = [f32; 5];
    }

    #[test]
    fn test_backward() {
        let t1: Tensor = Default::default();
        let _t1: Tensor = Default::default();

        let mut tape = GradientTape::default();
        tape.add_operation(move |g| {
            g.mut_gradient(&_t1).add(&[1.0; 5]);
        });
        let g = tape.backward(&t1);
        assert_eq!(g.gradient(&t1), &[1.0; 5]);
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
