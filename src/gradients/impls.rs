use super::structs::*;
use ndarray::prelude::*;
use std::ops::Index;

impl Gradient {
    pub(crate) fn without_tape(gradient_ref: GradientRef) -> Self {
        Self {
            gradient_ref,
            tape: None,
        }
    }
}

impl Index<GradientRef> for GradientTape {
    type Output = ArrayD<f32>;
    fn index(&self, gradient_ref: GradientRef) -> &Self::Output {
        &self.gradients[gradient_ref.index]
    }
}

impl GradientTape {
    pub fn new() -> Self {
        Self {
            derivatives: Vec::new(),
            gradients: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn scale(&mut self, scalar: f32) {
        for g in self.gradients.iter_mut() {
            *g *= scalar;
        }
    }

    pub(crate) fn store_derivative<D: Dimension>(&mut self, deriv: Array<f32, D>) -> DerivativeRef {
        let index = self.derivatives.len();
        self.derivatives.push(deriv.into_dyn());
        DerivativeRef { index }
    }

    fn allocate_gradient_ref<Sh: ShapeBuilder>(&mut self, shape: Sh) -> GradientRef {
        let index = self.gradients.len();
        self.gradients.push(Array::zeros(shape).into_dyn());
        GradientRef { index }
    }

    pub(crate) fn allocate_gradient<Sh: ShapeBuilder>(&mut self, shape: Sh) -> Gradient {
        let gradient_ref = self.allocate_gradient_ref(shape);
        Gradient::without_tape(gradient_ref)
    }

    pub(crate) fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    fn grad(&self, gradient_ref: GradientRef) -> &ArrayD<f32> {
        &self.gradients[gradient_ref.index]
    }

    fn deriv(&self, derivative_ref: DerivativeRef) -> &ArrayD<f32> {
        &self.derivatives[derivative_ref.index]
    }

    pub fn backward(&mut self, gradient_ref: GradientRef) {
        self.gradients[gradient_ref.index].fill(1.0);
        for operation in self.operations.iter().rev() {
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
                    OpType::Broadcast => {
                        let d_grad = self.deriv(op.parent_derivs[0]) * self.grad(op.result_grad);
                        self.gradients[op.parent_grads[0].index] += &d_grad;

                        let d_grad = self.deriv(op.parent_derivs[1])
                            * &self.grad(op.result_grad).sum_axis(Axis(0));
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
        }
    }
}
