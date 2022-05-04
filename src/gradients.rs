use super::array_ops::{AddElements, CountElements, MapElements, SubElements, ZeroElements};
use crate::prelude::{
    DivElements, FillElements, MulElements, ReduceElements, ReduceInnerElements, ScaleElements,
    ZipMapElements,
};
use std::collections::HashMap;

pub trait HasUniqueId {
    fn id(&self) -> usize;
}

pub trait HasNdArray {
    type ArrayType: 'static
        + Sized
        + Clone
        + ZipMapElements<Self::ArrayType>
        + MapElements
        + AddElements<Self::ArrayType>
        + SubElements<Self::ArrayType>
        + MulElements<Self::ArrayType>
        + DivElements<Self::ArrayType>
        + ZeroElements
        + CountElements
        + ReduceElements
        + ScaleElements
        + FillElements;

    fn data(&self) -> &Self::ArrayType;
    fn mut_data(&mut self) -> &mut Self::ArrayType;
}

pub struct GradientTape {
    grad_ref_by_id: HashMap<usize, GradientRef>,
    operations: Vec<Box<dyn FnOnce(&mut GradientTape) -> ()>>,
    gradients: Vec<Box<dyn std::any::Any>>,
}

impl std::fmt::Debug for GradientTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("grad_ref_by_id", &self.grad_ref_by_id)
            .field("gradients", &self.gradients)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GradientRef {
    index: usize,
}

impl Default for GradientTape {
    fn default() -> Self {
        Self {
            grad_ref_by_id: HashMap::new(),
            gradients: Vec::new(),
            operations: Vec::new(),
        }
    }
}

impl GradientTape {
    fn make_or_get_grad_ref<T: HasUniqueId + HasNdArray>(&mut self, t: &T) -> GradientRef {
        match self.grad_ref_by_id.get(&t.id()) {
            Some(grad_ref) => *grad_ref,
            None => {
                let index = self.gradients.len();
                let grad_ref = GradientRef { index };
                self.gradients.push(Box::new(T::ArrayType::ZEROS));
                self.grad_ref_by_id.insert(t.id(), grad_ref);
                grad_ref
            }
        }
    }

    pub fn gradient<T: HasUniqueId + HasNdArray>(&self, t: &T) -> &T::ArrayType {
        let grad_ref = self.grad_ref_by_id.get(&t.id()).unwrap();
        self.gradients[grad_ref.index].downcast_ref().unwrap()
    }

    pub fn mut_gradient<T: HasUniqueId + HasNdArray>(&mut self, t: &T) -> &mut T::ArrayType {
        let grad_ref = self.make_or_get_grad_ref(t);
        self.gradients[grad_ref.index].downcast_mut().unwrap()
    }

    pub(crate) fn add_operation<F: 'static + FnOnce(&mut GradientTape) -> ()>(
        &mut self,
        operation: F,
    ) {
        self.operations.insert(0, Box::new(operation));
    }

    pub fn backward<T: HasUniqueId + HasNdArray>(&mut self, t: &T) {
        self.mut_gradient(t).map_assign_elems(|v| *v = 1.0);
        let ops: Vec<Box<dyn FnOnce(&mut GradientTape)>> = self.operations.drain(..).collect();
        for operation in ops {
            (operation)(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array_ops::AddElements;

    use super::*;

    #[derive(Default)]
    struct Tensor {
        id: usize,
        data: [f32; 5],
    }

    impl HasUniqueId for Tensor {
        fn id(&self) -> usize {
            self.id
        }
    }

    impl HasNdArray for Tensor {
        type ArrayType = [f32; 5];
        fn data(&self) -> &Self::ArrayType {
            &self.data
        }
        fn mut_data(&mut self) -> &mut Self::ArrayType {
            &mut self.data
        }
    }

    #[test]
    fn test_backward() {
        let t1: Tensor = Default::default();

        // let mut tape = GradientTape::default();
        // tape.add_operation(|tape| {
        //     tape.mut_gradient(&t1).add(&[1.0; 5]);
        // });
        // assert_eq!(tape.gradient(&t1), &[0.0; 5]);
        // tape.backward(&t1);
        // assert_eq!(tape.gradient(&t1), &[1.0; 5]);
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
